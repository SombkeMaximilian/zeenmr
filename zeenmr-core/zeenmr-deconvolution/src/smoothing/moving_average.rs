use crate::smoothing::Smooth;
use num_traits::Float;
use std::borrow::Cow;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Moving average filter.
///
/// Smooths a sequence of values by averaging them over a sliding window.
///
/// # Edge Handling
///
/// The window is centered around the current value. To handle the edges of the
/// input sequence, the window starts at half its size (rounded up) at the left
/// edge and grows to its full size when there are enough values to the left of
/// the current value. For example, with a window size of 5 (window center
/// marked by `x`, extent marked by `-`):
///
/// | Index  | 0   | 1   | 2   | 3   | 4   | 5   | 6   |
/// | ------ | --- | --- | --- | --- | --- | --- | --- |
/// | Step 1 | x   | -   | -   |     |     |     |     |
/// | Step 2 | -   | x   | -   | -   |     |     |     |
/// | Step 3 | -   | -   | x   | -   | -   |     |     |
/// | Step 4 |     | -   | -   | x   | -   | -   |     |
/// | Step 5 |     |     | -   | -   | x   | -   | -   |
/// | Step 6 |     |     |     | -   | -   | x   | -   |
/// | Step 7 |     |     |     |     | -   | -   | x   |
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MovingAverage {
    /// Number of iterations to apply the filter.
    pub iterations: usize,
    /// Size of the sliding window.
    pub window_size: usize,
}

impl<T> Smooth<T> for MovingAverage
where
    T: Clone + Float,
{
    fn smooth<'a>(&self, data: &'a [T]) -> Cow<'a, [T]> {
        if data.len() < 2
            || data.len() < self.window_size
            || self.iterations == 0
            || self.window_size <= 1
        {
            return data.into();
        }

        let mut data = data.to_vec();
        let half_window = self.window_size / 2;
        let len = data.len();
        let full_div = T::one()
            / T::from(self.window_size).expect("conversion from usize to T must never fail");
        for _ in 0..self.iterations {
            let mut sum = data
                .iter()
                .take(half_window)
                .fold(T::zero(), |acc, x| acc + *x);
            for curr in 0..half_window {
                sum = sum + data[curr + half_window];
                data[curr] = sum
                    / T::from(half_window + curr + 1)
                        .expect("conversion from usize to T must never fail");
            }
            for curr in half_window..(len - half_window) {
                sum = sum + data[curr + half_window] - data[curr - half_window];
                data[curr] = sum * full_div;
            }
            for curr in (len - half_window)..len {
                sum = sum - data[curr - half_window];
                data[curr] = sum
                    / T::from(len - curr + half_window - 1)
                        .expect("conversion from usize to T must never fail");
            }
        }

        data.into()
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
    /// Creates a new `MovingAverage` filter.
    pub fn new(iterations: usize, window_size: usize) -> Self {
        Self {
            iterations,
            window_size,
        }
    }
}
