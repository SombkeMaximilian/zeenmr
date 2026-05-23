use crate::smoothing::Smooth;
use std::borrow::Borrow;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Moving average filter that smooths a sequence of values by averaging them
/// over a sliding window.
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

impl Smooth for MovingAverage {
    fn smooth<I>(&self, data: I) -> Vec<f64>
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        let mut data = data
            .into_iter()
            .map(|value| *(value.borrow()))
            .collect::<Vec<f64>>();
        let half_window = self.window_size / 2;
        let len = data.len();
        let full_div = 1_f64 / (self.window_size as f64);
        for _ in 0..self.iterations {
            let mut sum = data.iter().take(half_window).sum::<f64>();
            for curr in 0..half_window {
                sum += data[curr + half_window];
                data[curr] = sum / ((half_window + curr + 1) as f64)
            }
            for curr in half_window..(len - half_window) {
                sum += data[curr + half_window] - data[curr - half_window];
                data[curr] = sum * full_div;
            }
            for curr in (len - half_window)..len {
                sum -= data[curr - half_window];
                data[curr] = sum / ((len - curr + half_window) as f64)
            }
        }

        data
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
    /// Creates a new `MovingAverage` filter with the specified number of
    /// iterations and window size.
    pub fn new(iterations: usize, window_size: usize) -> Self {
        Self {
            iterations,
            window_size,
        }
    }
}
