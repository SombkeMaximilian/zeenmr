use crate::smoothing::Smooth;
use num_traits::Float;
use zeenmr_spectrum::dimension::DimIndex;
use zeenmr_spectrum::intensity_array::{ArrayViewMut1D, index};
use zeenmr_util::fma;

mod cache_topology {
    //! Contains cache size constants in bytes.

    include!(concat!(env!("OUT_DIR"), "/cache.rs"));
}

/// Chunk size in elements.
const CHUNK: usize = 16;

/// Edge handling logic.
///
/// # Note for Implementors
///
/// It can be assumed that `values.len() >= coefficients.len()`, and that
/// `coefficients` is not empty. The returned `Vec<T>` contains the concatenated
/// head and tail corrections, split at the returned index. Both parts must
/// cover the respective edges exactly, otherwise the smoothing implementation
/// of [`CoefficientWindow`] will panic.
pub trait CorrectEdges {
    /// Applies the edge correction.
    fn correct_edges<T>(&self, values: &[T], coefficients: &[T]) -> (Vec<T>, usize)
    where
        T: Float;
}

/// Applies the edge correction by mirror the first and last few values.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct MirrorCorrection;

impl CorrectEdges for MirrorCorrection {
    fn correct_edges<T>(&self, values: &[T], coefficients: &[T]) -> (Vec<T>, usize)
    where
        T: Float,
    {
        let len = values.len();
        let kernel_size = coefficients.len() - 1;
        let head = kernel_size / 2;
        let tail = kernel_size.div_ceil(2);
        let extension_len = kernel_size + head.max(tail);
        let mut edge_scratch = vec![T::zero(); extension_len + head + tail];
        let (head_values, rest) = edge_scratch.split_at_mut(head);
        let (tail_values, extension) = rest.split_at_mut(tail);
        let (mirror, body) = extension.split_at_mut(head);
        mirror
            .iter_mut()
            .zip(values[..head].iter().rev())
            .for_each(|(value, &new)| {
                *value = new;
            });
        body[..kernel_size].copy_from_slice(&values[..kernel_size]);
        head_values
            .iter_mut()
            .zip(extension[..(head + kernel_size)].windows(kernel_size + 1))
            .for_each(|(value, window)| {
                *value = coefficients
                    .iter()
                    .zip(window.iter())
                    .fold(T::zero(), |acc, (&weight, &new)| fma(weight, new, acc));
            });
        extension[..kernel_size].copy_from_slice(&values[(len - kernel_size)..]);
        extension[kernel_size..(kernel_size + tail)]
            .iter_mut()
            .zip(values[(len - tail)..].iter().rev())
            .for_each(|(value, &new)| {
                *value = new;
            });
        tail_values
            .iter_mut()
            .zip(extension[..(kernel_size + tail)].windows(kernel_size + 1))
            .for_each(|(value, window)| {
                *value = coefficients
                    .iter()
                    .zip(window.iter())
                    .fold(T::zero(), |acc, (&weight, &new)| fma(weight, new, acc));
            });
        edge_scratch.truncate(head + tail);

        (edge_scratch, head)
    }
}

/// Coefficient window filter with edge correction.
#[derive(Clone, PartialEq, Debug)]
pub struct CoefficientWindow<C> {
    /// Coefficients to apply to each window.
    coefficients: Box<[f64]>,
    /// Edge correction to handle missing values for the window.
    correction: C,
}

impl<T, C> Smooth<T> for CoefficientWindow<C>
where
    T: Float,
    C: CorrectEdges,
{
    type Error = std::convert::Infallible;

    fn smooth_in_place(&self, mut data: ArrayViewMut1D<T>) -> Result<(), Self::Error> {
        let mut lane = data
            .lane_at_mut(DimIndex(0), &index([0]))
            .expect("1D array always has a first dimension");
        match lane.as_mut_slice() {
            Some(slice) => self.smooth_slice(slice),
            None => {
                let mut values = lane.iter().copied().collect::<Vec<T>>();
                self.smooth_slice(&mut values);
                lane.iter_mut()
                    .zip(values)
                    .for_each(|(value, new)| *value = new);
            }
        }

        Ok(())
    }
}

impl<C> CoefficientWindow<C> {
    /// Creates a new coefficient window filter.
    pub fn new(coefficients: Box<[f64]>, correction: C) -> Self {
        Self {
            coefficients,
            correction,
        }
    }
}

impl<C> CoefficientWindow<C>
where
    C: CorrectEdges,
{
    /// Applies the coefficient window filter to a slice in-place.
    fn smooth_slice<T>(&self, values: &mut [T])
    where
        T: Float,
    {
        let len = values.len();
        let kernel_size = self.coefficients.len() - 1;

        if len < kernel_size {
            return;
        }

        let coefficients = self
            .coefficients
            .iter()
            .map(|&weight| T::from(weight).expect("conversion from f64 to T must never fail"))
            .collect::<Box<[T]>>();
        let head = kernel_size / 2;
        let tail = kernel_size.div_ceil(2);
        let (edges, edge_split) = self
            .correction
            .correct_edges(values, &coefficients);

        assert_eq!(edge_split, head);
        assert_eq!(edges.len(), head + tail);

        let tile_len = tile_len::<T>();
        let mid_len = len - head - tail;
        let mut tile = vec![T::zero(); tile_len + kernel_size];
        tile[..kernel_size].copy_from_slice(&values[..kernel_size]);
        let mut done = 0;
        while done < mid_len {
            let curr_len = tile_len.min(mid_len - done);
            tile[kernel_size..(kernel_size + curr_len)].copy_from_slice(
                &values[(done + kernel_size)..(done + kernel_size + curr_len)]
            );
            let mut chunks =
                values[(head + done)..(head + done + curr_len)].chunks_exact_mut(CHUNK);
            let mut chunk_offset = 0;
            for out_chunk in &mut chunks {
                let mut sum = [T::zero(); CHUNK];
                for (shift, &weight) in coefficients.iter().enumerate() {
                    sum.iter_mut()
                        .zip(tile[(chunk_offset + shift)..(chunk_offset + shift + CHUNK)].iter())
                        .for_each(|(acc, &new)| *acc = fma(weight, new, *acc));
                }
                out_chunk.copy_from_slice(&sum);
                chunk_offset += CHUNK;
            }
            for (shift, value) in chunks.into_remainder().iter_mut().enumerate() {
                *value = coefficients
                    .iter()
                    .zip(tile[(chunk_offset + shift)..].iter())
                    .fold(T::zero(), |acc, (&weight, &new)| fma(weight, new, acc));
            }
            tile.copy_within(curr_len..(curr_len + kernel_size), 0);
            done += curr_len;
        }

        let (head_values, tail_values) = edges.split_at(edge_split);
        values[..head].copy_from_slice(head_values);
        values[(len - tail)..].copy_from_slice(tail_values);
    }
}

/// Computes tile length that optimally uses the L1 cache.
const fn tile_len<T>() -> usize {
    let t = size_of::<T>();
    let usable = cache_topology::L1D_PER_THREAD.saturating_mul(4) / 5;
    let tile = usable / (2 * t) / CHUNK * CHUNK;

    if tile < 4 * CHUNK { 4 * CHUNK } else { tile }
}
