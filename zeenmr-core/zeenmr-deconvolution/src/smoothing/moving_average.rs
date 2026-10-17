use crate::smoothing::{CoefficientWindow, MirrorCorrection, Smooth};
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
    convolution: Option<CoefficientWindow<MirrorCorrection>>,
}

impl<T> Smooth<T> for MovingAverage
where
    T: Float,
{
    type Error = std::convert::Infallible;

    fn smooth_in_place(&self, mut data: ArrayViewMut1D<T>) -> Result<(), Self::Error> {
        if data.len() < 2 || data.len() < self.window || self.iterations == 0 || self.window == 1 {
            return Ok(());
        }

        if let Some(convolution) = self.convolution.as_ref() {
            convolution
                .smooth_in_place(data)
                .expect("coefficient convolution is infallible");
        } else {
            let mut lane = data
                .lane_at_mut(DimIndex(0), &index([0]))
                .expect("1D array always has a first dimension");
            match lane.as_mut_slice() {
                Some(values) => self.iterative_fallback(values),
                None => {
                    let mut values = lane.iter().copied().collect::<Vec<T>>();
                    self.iterative_fallback(&mut values);
                    lane.iter_mut()
                        .zip(values)
                        .for_each(|(value, new)| *value = new);
                }
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
        let convolution = Self::convolution_coefficients(iterations, window)
            .map(|coefficients| CoefficientWindow::new(coefficients, MirrorCorrection));

        Self {
            iterations,
            window,
            convolution,
        }
    }

    /// Iteratively computes the moving average, accumulating it into `values`.
    ///
    /// This is around 2-5x slower than the convolution.
    fn iterative_fallback<T>(&self, values: &mut [T])
    where
        T: Float,
    {
        let len = values.len();
        let back = (self.window - 1) / 2;
        let forward = self.window / 2;
        let last_mid = len - self.window + 1;
        let div = T::from(self.window)
            .expect("conversion from usize to T must never fail")
            .recip();
        let mut scratch = vec![T::zero(); len];
        for _ in 0..self.iterations {
            let mid = &mut scratch[back..(back + last_mid)];
            let mut chunks = mid.chunks_exact_mut(CHUNK);
            let mut chunk_offset = 0;
            for out_chunk in &mut chunks {
                let mut sum = [T::zero(); CHUNK];
                for shift in 0..self.window {
                    sum.iter_mut()
                        .zip(values[(chunk_offset + shift)..(chunk_offset + shift + CHUNK)].iter())
                        .for_each(|(acc, &new)| *acc = fma(div, new, *acc));
                }
                out_chunk.copy_from_slice(&sum);
                chunk_offset += CHUNK;
            }
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                for (shift, value) in rem.iter_mut().enumerate() {
                    *value = values[(chunk_offset + shift)..(chunk_offset + shift + self.window)]
                        .iter()
                        .fold(T::zero(), |acc, &new| fma(div, new, acc));
                }
            }
            for center in (0..back).chain((back + last_mid)..len) {
                let left = center.saturating_sub(back);
                let right = (center + forward + 1).min(len);
                let div = T::from(right - left)
                    .expect("conversion from usize to T must not fail")
                    .recip();
                scratch[center] = values[left..right]
                    .iter()
                    .fold(T::zero(), |acc, &new| fma(div, new, acc));
            }
            values.swap_with_slice(&mut scratch);
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
            let new_half_len = new_full_len.div_ceil(2);
            let count_at = |idx: usize| half_counts[idx.min(full_len - 1 - idx)];
            let mut new_half = Vec::with_capacity(new_half_len);
            let mut window_sum = 0_u128;
            for index in 0..new_half_len {
                if index < full_len {
                    window_sum += count_at(index);
                }
                if index >= window {
                    window_sum -= count_at(index - window);
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
