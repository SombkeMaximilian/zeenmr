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

/// Moving average filter.
///
/// Smooths a sequence of values by averaging them over a sliding window.
#[derive(Clone, PartialEq, Debug)]
pub struct MovingAverage {
    /// Number of iterations to apply the filter.
    iterations: usize,
    /// Size of the sliding window.
    window: usize,
    /// Convoluted coefficients, if they can be computed.
    coefficients: Option<Box<[f64]>>,
}

impl<T> Smooth<T> for MovingAverage
where
    T: Float,
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

impl Default for MovingAverage {
    fn default() -> Self {
        Self::new(3, 3)
    }
}

impl MovingAverage {
    /// Creates a new moving average filter.
    pub fn new(iterations: usize, window: usize) -> Self {
        Self {
            iterations,
            window,
            coefficients: Self::convolution_coefficients(iterations, window),
        }
    }

    /// Smooths a slice in-place.
    fn smooth_slice<T>(&self, values: &mut [T])
    where
        T: Float,
    {
        let len = values.len();
        if len < 2 || len < self.window || self.iterations == 0 || self.window == 1 {
            return;
        }
        let kernel_size = self.iterations * (self.window - 1);
        if len >= kernel_size {
            if self.is_convolution() {
                self.convolve(values, kernel_size);
            } else {
                let mut scratch = vec![T::zero(); len];
                self.iterative_fallback(values, &mut scratch);
            }
        } else {
            let mut scratch = vec![T::zero(); len];
            self.iterative_small(values, &mut scratch);
        };
    }

    /// Returns `true` if the moving average filter can be applied as one-pass
    /// convolution.
    fn is_convolution(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Computes the moving average in a single pass using the coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the convoluted coefficients weren't calculated.
    fn convolve<T>(&self, values: &mut [T], kernel_size: usize)
    where
        T: Float,
    {
        let coefficients = self
            .coefficients
            .as_deref()
            .expect("coefficients should be Some")
            .iter()
            .map(|&weight| T::from(weight).expect("conversion from f64 to T must never fail"))
            .collect::<Box<[T]>>();
        let len = values.len();
        let head = self.iterations * ((self.window - 1) / 2);
        let tail = self.iterations * (self.window / 2);

        let edge_len = kernel_size.max(self.window + 1).min(len);
        let mut edge_scratch = vec![T::zero(); 2 * edge_len + head + tail];
        let (edge_curr, rest) = edge_scratch.split_at_mut(edge_len);
        let (edge_next, rest) = rest.split_at_mut(edge_len);
        let (head_values, tail_values) = rest.split_at_mut(head);
        edge_curr.copy_from_slice(&values[..edge_len]);
        self.iterative_small(edge_curr, edge_next);
        head_values.copy_from_slice(&edge_curr[..head]);
        edge_curr.copy_from_slice(&values[(len - edge_len)..]);
        self.iterative_small(edge_curr, edge_next);
        tail_values.copy_from_slice(&edge_curr[(edge_len - tail)..]);

        let tile_len = tile_len::<T>();
        let mid_len = len - head - tail;
        let mut tile = vec![T::zero(); tile_len + kernel_size];
        let first = tile_len.min(mid_len) + kernel_size;
        tile[..first].copy_from_slice(&values[..first]);
        let mut done = 0;
        while done < mid_len {
            let tile_len = tile_len.min(mid_len - done);
            let mut chunks =
                values[(head + done)..(head + done + tile_len)].chunks_exact_mut(CHUNK);
            let mut chunk_offset = 0;
            for out_chunk in &mut chunks {
                let mut sum = [T::zero(); CHUNK];
                for (shift, &weight) in coefficients.iter().enumerate() {
                    sum.iter_mut()
                        .zip(&tile[(chunk_offset + shift)..(chunk_offset + shift + CHUNK)])
                        .for_each(|(acc, &new)| *acc = fma(weight, new, *acc));
                }
                out_chunk.copy_from_slice(&sum);
                chunk_offset += CHUNK;
            }
            for (shift, value) in chunks.into_remainder().iter_mut().enumerate() {
                *value = coefficients
                    .iter()
                    .zip(&tile[(chunk_offset + shift)..])
                    .fold(T::zero(), |acc, (&weight, &new)| fma(weight, new, acc));
            }
            done += tile_len;
            if done < mid_len {
                let next_tile_len = tile_len.min(mid_len - done);
                tile.copy_within(tile_len..(tile_len + kernel_size), 0);
                tile[kernel_size..(kernel_size + next_tile_len)].copy_from_slice(
                    &values[(done + kernel_size)..(done + kernel_size + next_tile_len)],
                );
            }
        }
        values[..head].copy_from_slice(head_values);
        values[(len - tail)..].copy_from_slice(tail_values);
    }

    /// Iteratively computes the moving average, accumulating it into `values`.
    ///
    /// This is around 2-5x slower than the convoluted coefficients.
    fn iterative_fallback<T>(&self, values: &mut [T], scratch: &mut [T])
    where
        T: Float,
    {
        let len = values.len();
        let back = (self.window - 1) / 2;
        let forward = self.window / 2;
        let last_mid = len - self.window + 1;
        let full_div = T::from(self.window)
            .expect("conversion from usize to T must never fail")
            .recip();
        for _ in 0..self.iterations {
            let mid = &mut scratch[back..(back + last_mid)];
            let mut chunks = mid.chunks_exact_mut(CHUNK);
            let mut chunk_offset = 0;
            for out_chunk in &mut chunks {
                let mut sum = [T::zero(); CHUNK];
                for shift in 0..self.window {
                    sum.iter_mut()
                        .zip(values[(chunk_offset + shift)..(chunk_offset + shift + CHUNK)].iter())
                        .for_each(|(acc, &new)| *acc = *acc + new);
                }
                out_chunk
                    .iter_mut()
                    .zip(sum)
                    .for_each(|(value, acc)| *value = acc * full_div);
                chunk_offset += CHUNK;
            }
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let rem_len = rem.len();
                rem.copy_from_slice(&values[chunk_offset..(chunk_offset + rem_len)]);
                for shift in 1..self.window {
                    rem.iter_mut()
                        .zip(&values[(chunk_offset + shift)..(chunk_offset + shift + rem_len)])
                        .for_each(|(value, &new)| *value = *value + new);
                }
                rem.iter_mut()
                    .for_each(|value| *value = *value * full_div);
            }
            for center in (0..back).chain((back + last_mid)..len) {
                let left = center.saturating_sub(back);
                let right = (center + forward + 1).min(len);
                let sum = values[left..right]
                    .iter()
                    .fold(T::zero(), |acc, &new| acc + new);
                scratch[center] =
                    sum / T::from(right - left).expect("conversion from usize to T must not fail");
            }
            values.swap_with_slice(scratch);
        }
    }

    /// Iteratively computes the moving average, accumulating it into `values`.
    ///
    /// Faster for small inputs.
    fn iterative_small<T>(&self, values: &mut [T], scratch: &mut [T])
    where
        T: Float,
    {
        let len = values.len();
        let back = (self.window - 1) / 2;
        let forward = self.window / 2;
        let last_mid = len - self.window + 1;
        let full_div = T::from(self.window)
            .expect("conversion from usize to T must never fail")
            .recip();
        for _ in 0..self.iterations {
            let mid = &mut scratch[back..(back + last_mid)];
            mid.copy_from_slice(&values[..last_mid]);
            for shift in 1..self.window {
                mid.iter_mut()
                    .zip(values[shift..].iter())
                    .for_each(|(value, &new)| {
                        *value = *value + new;
                    });
            }
            mid.iter_mut()
                .for_each(|value| *value = *value * full_div);
            for center in (0..back).chain((back + last_mid)..len) {
                let left = center.saturating_sub(back);
                let right = (center + forward + 1).min(len);
                let sum = values[left..right]
                    .iter()
                    .fold(T::zero(), |acc, &new| acc + new);
                scratch[center] =
                    sum / T::from(right - left).expect("conversion from usize to T must not fail");
            }
            values.swap_with_slice(scratch);
        }
    }

    /// Computes the convoluted coefficients to do a single pass with more
    /// precise results.
    ///
    /// Returns `None` if `window^iterations` overflows.
    fn convolution_coefficients(iterations: usize, window: usize) -> Option<Box<[f64]>> {
        let total = (window as u128).checked_pow(iterations as u32)?;
        let mut half_counts = vec![1_u128];
        let mut full_len = 1;
        for _ in 0..iterations {
            let new_full_len = full_len + window - 1;
            let new_half_len = (new_full_len + 1) / 2;
            let count_at = |idx: usize| half_counts[idx.min(full_len - 1 - idx)];
            let mut new_half = Vec::with_capacity(new_half_len);
            let mut window_sum = 0_u128;
            for idx in 0..new_half_len {
                if idx < full_len {
                    window_sum += count_at(idx);
                }
                if idx >= window {
                    window_sum -= count_at(idx - window);
                }
                new_half.push(window_sum);
            }
            half_counts = new_half;
            full_len = new_full_len;
        }
        let denominator = total as f64;

        Some(
            (0..full_len)
                .map(|idx| half_counts[idx.min(full_len - 1 - idx)] as f64 / denominator)
                .collect(),
        )
    }
}

/// Computes tile length that optimally uses the L1 cache.
const fn tile_len<T>() -> usize {
    let t = size_of::<T>();
    let usable = cache_topology::L1D_PER_THREAD.saturating_mul(4) / 5;
    let tile = usable / (2 * t) / CHUNK * CHUNK;

    if tile < 4 * CHUNK { 4 * CHUNK } else { tile }
}
