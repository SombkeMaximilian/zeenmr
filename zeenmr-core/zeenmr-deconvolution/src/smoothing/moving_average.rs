use crate::smoothing::{CircularBuffer, Smooth};
use num_traits::Float;
use std::borrow::Borrow;

#[derive(Copy, Clone, Debug)]
pub struct MovingAverage {
    /// Number of iterations to apply the filter.
    iterations: usize,
    /// Size of the sliding window.
    window_size: usize,
}

impl<F> Smooth<F> for MovingAverage
where
    F: Copy + Float,
{
    fn smooth<I>(&self, data: I) -> impl Iterator<Item = F>
    where
        I: IntoIterator,
        I::Item: Borrow<F>,
    {
        let mut data = data
            .into_iter()
            .map(|value| *value.borrow())
            .collect::<Vec<F>>();
        let mut cache = CircularBuffer::<F>::new(self.window_size);
        let half_window = self.window_size / 2;
        let len = data.len();
        for _ in 0..self.iterations {
            let mut div = F::one();
            let mut sum = F::zero();
            for value in data.iter().take(half_window) {
                cache.push(*value);
                sum = sum + *value;
            }
            for i in 0..(len - half_window) {
                sum = sum + data[i + half_window];
                if let Some(popped) = cache.push(data[i + half_window]) {
                    sum = sum - popped;
                } else {
                    div = F::one() / F::from(cache.len()).unwrap();
                };
                data[i] = sum * div;
            }
            for value in data[(len - half_window)..].iter_mut() {
                if let Some(popped) = cache.pop() {
                    sum = sum - popped;
                    div = F::one() / F::from(cache.len()).unwrap();
                    *value = sum * div;
                }
            }
            cache.clear();
        }

        data.into_iter()
    }
}

impl Default for MovingAverage {
    fn default() -> Self {
        Self {
            iterations: 3,
            window_size: 3,
        }
    }
}

impl MovingAverage {
    pub fn new(iterations: usize, window_size: usize) -> Self {
        Self {
            iterations,
            window_size,
        }
    }
}
