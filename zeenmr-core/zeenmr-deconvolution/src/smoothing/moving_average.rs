use crate::smoothing::{CircularBuffer, Smooth};
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
    type Settings = Self;

    fn settings(&self) -> Self::Settings {
        *self
    }

    fn smooth<I>(&self, data: I) -> Vec<f64>
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        let mut data = data
            .into_iter()
            .map(|value| *value.borrow())
            .collect::<Vec<f64>>();
        let mut cache = CircularBuffer::<f64>::new(self.window_size);
        let half_window = self.window_size / 2;
        let len = data.len();
        for _ in 0..self.iterations {
            let mut div = 1.0;
            let mut sum = 0.0;
            for value in data.iter().take(half_window) {
                cache.push(*value);
                sum = sum + *value;
            }
            for i in 0..(len - half_window) {
                sum = sum + data[i + half_window];
                if let Some(popped) = cache.push(data[i + half_window]) {
                    sum = sum - popped;
                } else {
                    div = 1.0 / cache.len() as f64;
                };
                data[i] = sum * div;
            }
            for value in data[(len - half_window)..].iter_mut() {
                if let Some(popped) = cache.pop() {
                    sum = sum - popped;
                    div = 1.0 / cache.len() as f64;
                    *value = sum * div;
                }
            }
            cache.clear();
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
