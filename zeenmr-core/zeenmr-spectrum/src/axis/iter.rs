//! Iterator types for axes.

use crate::iter::SplitAt;
use num_traits::Float;
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use crate::iter::Par;

/// Parallel iterator over evenly spaced elements spanning an axis.
#[cfg(feature = "rayon")]
pub type ParAxisValues<T> = Par<AxisValues<T>>;

/// Iterator over evenly spaced elements spanning an axis.
#[derive(Clone, Debug)]
pub struct AxisValues<T> {
    /// Start of the range.
    start: T,
    /// Signed step size.
    step: T,
    /// Next element index from the front.
    front: usize,
    /// Next element index from the back + 1.
    back: usize,
}

impl<T> Iterator for AxisValues<T>
where
    T: Float,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = T::from(self.front).expect("conversion from usize to T must never fail");
            self.front += 1;

            Some(self.start + self.step * curr)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;

        (remaining, Some(remaining))
    }
}

impl<T> DoubleEndedIterator for AxisValues<T>
where
    T: Float,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            let curr = T::from(self.back).expect("conversion from usize to T must never fail");

            Some(self.start + self.step * curr)
        } else {
            None
        }
    }
}

impl<T> ExactSizeIterator for AxisValues<T> where T: Float {}

impl<T> FusedIterator for AxisValues<T> where T: Float {}

// SAFETY: setting left.back and right.front to `self.front + index` or
// `self.back`, whichever is smaller, achieves the disjoint split required by
// `SplitAt`.
unsafe impl<T> SplitAt for AxisValues<T>
where
    T: Clone,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = (self.front + index).max(self.back);
        let left = Self {
            start: self.start.clone(),
            step: self.step.clone(),
            front: self.front,
            back: mid,
        };
        let right = Self {
            start: self.start,
            step: self.step,
            front: mid,
            back: self.back,
        };

        (left, right)
    }
}

impl<T> AxisValues<T> {
    /// Creates a new `AxisValues`.
    ///
    /// Prefer using the methods on the concrete axis types.
    pub fn new(start: T, step: T, len: usize) -> Self {
        AxisValues {
            start,
            step,
            front: 0,
            back: len,
        }
    }
}
