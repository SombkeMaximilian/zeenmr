//! Iterator types for axes.

use num_traits::Float;
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

/// Parallel iterator over evenly spaced elements spanning an axis.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParAxisValues<T>(AxisValues<T>);

#[cfg(feature = "rayon")]
impl<T> ParallelIterator for ParAxisValues<T>
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
impl<T> IndexedParallelIterator for ParAxisValues<T>
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
        callback.callback(AxisValuesProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<T> ParAxisValues<T>
where
    T: Float + Send,
{
    /// Creates a new `ParAxisValues`.
    ///
    /// Prefer using the methods on the concrete axis types.
    pub fn new(start: T, step: T, len: usize) -> Self {
        Self(AxisValues::new(start, step, len))
    }
}

/// Producer for [`ParAxisValues`].
#[cfg(feature = "rayon")]
struct AxisValuesProducer<T>(AxisValues<T>);

#[cfg(feature = "rayon")]
impl<T> Producer for AxisValuesProducer<T>
where
    T: Float + Send,
{
    type Item = T;
    type IntoIter = AxisValues<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = AxisValues {
            back: mid,
            ..self.0
        };
        let right = AxisValues {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}
