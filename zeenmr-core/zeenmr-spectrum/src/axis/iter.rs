//! Iterator types for axes.

use num_traits::Float;
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Iterator over evenly spaced elements spanning an axis.
#[derive(Clone, Debug)]
pub struct AxisIter<T> {
    /// Start of the range.
    start: T,
    /// Signed step size.
    step: T,
    /// Next element index from the front.
    front: usize,
    /// Next element index from the back + 1.
    back: usize,
}

impl<T> Iterator for AxisIter<T>
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

impl<T> DoubleEndedIterator for AxisIter<T>
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

impl<T> ExactSizeIterator for AxisIter<T> where T: Float {}

impl<T> FusedIterator for AxisIter<T> where T: Float {}

impl<T> AxisIter<T> {
    /// Creates a new `AxisIter`.
    ///
    /// Prefer using the methods on the concrete axis types.
    pub fn new(start: T, step: T, len: usize) -> Self {
        AxisIter {
            start,
            step,
            front: 0,
            back: len,
        }
    }
}

/// Parallel iterator over evenly spaced elements spanning an axis.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParAxisIter<T>(AxisIter<T>);

#[cfg(feature = "rayon")]
impl<T> ParallelIterator for ParAxisIter<T>
where
    T: Float + Send,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

#[cfg(feature = "rayon")]
impl<T> IndexedParallelIterator for ParAxisIter<T>
where
    T: Float + Send,
{
    fn len(&self) -> usize {
        self.0.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(AxisIterProducer(self.0))
    }
}

impl<T> ParAxisIter<T>
where
    T: Float + Send,
{
    /// Creates a new `ParAxisIter`.
    ///
    /// Prefer using the methods on the concrete axis types.
    pub fn new(start: T, step: T, len: usize) -> Self {
        Self(AxisIter::new(start, step, len))
    }
}

/// Producer for [`ParAxisIter`].
#[cfg(feature = "rayon")]
struct AxisIterProducer<T>(AxisIter<T>);

#[cfg(feature = "rayon")]
impl<T> Producer for AxisIterProducer<T>
where
    T: Float + Send,
{
    type Item = T;
    type IntoIter = AxisIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = AxisIter {
            back: mid,
            ..self.0
        };
        let right = AxisIter {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}
