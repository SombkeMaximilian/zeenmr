use crate::intensity_array::{ArrayIndex, Dimension, Shape};
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Iterator over the multidimensional indices of a shape.
///
/// Yields indices in lexicographic order: the last dimension varies fastest.
#[derive(Clone, Debug)]
pub struct Indices<D> {
    /// Underlying shape.
    shape: Shape<D>,
    /// Next linear index from the front.
    front: usize,
    /// Next linear index from the back.
    back: usize,
    /// Index at `front`, if already computed.
    front_cached: Option<ArrayIndex<D>>,
    /// Index at `back - 1`, if already computed.
    back_cached: Option<ArrayIndex<D>>,
}

impl<D> Iterator for Indices<D>
where
    D: Dimension,
{
    type Item = ArrayIndex<D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let index = self
                .front_cached
                .get_or_insert_with(|| ArrayIndex::linear_in_shape(self.front, &self.shape));
            let curr = index.clone();
            index.increment_lexicographic(self.shape.as_slice());
            self.front += 1;

            Some(curr)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;

        (remaining, Some(remaining))
    }
}

impl<D> DoubleEndedIterator for Indices<D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            let index = self
                .back_cached
                .get_or_insert_with(|| ArrayIndex::linear_in_shape(self.back, &self.shape));
            let curr = index.clone();
            // wraps to zero on the last element, which `front == back` then
            // makes unreachable
            index.decrement_lexicographic(self.shape.as_slice());

            Some(curr)
        } else {
            None
        }
    }
}

impl<D> ExactSizeIterator for Indices<D> where D: Dimension {}

impl<D> FusedIterator for Indices<D> where D: Dimension {}

impl<D> Indices<D>
where
    D: Dimension,
{
    /// Creates a lexicographic index iterator.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    ///
    /// Prefer [`Shape::indices_lexicographic`].
    pub fn new(shape: Shape<D>) -> Option<Self> {
        let back = shape.product_checked()?;

        Some(Self {
            shape,
            front: 0,
            back,
            front_cached: None,
            back_cached: None,
        })
    }
}

/// Parallel iterator over the multidimensional indices of a shape.
///
/// Yields indices in lexicographic order: the last dimension varies fastest.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParIndices<D>(Indices<D>);

#[cfg(feature = "rayon")]
impl<D> ParallelIterator for ParIndices<D>
where
    D: Dimension,
{
    type Item = ArrayIndex<D>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.0.len())
    }
}

#[cfg(feature = "rayon")]
impl<D> IndexedParallelIterator for ParIndices<D>
where
    D: Dimension,
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
        callback.callback(IndicesProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<D> ParIndices<D>
where
    D: Dimension,
{
    /// Creates a parallel lexicographic index iterator.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    ///
    /// Prefer [`Shape::par_indices_lexicographic`].
    pub fn new(shape: Shape<D>) -> Option<Self> {
        Some(Self(Indices::new(shape)?))
    }
}

/// Producer for [`ParIndices`].
#[cfg(feature = "rayon")]
struct IndicesProducer<D>(Indices<D>);

#[cfg(feature = "rayon")]
impl<D> Producer for IndicesProducer<D>
where
    D: Dimension,
{
    type Item = ArrayIndex<D>;
    type IntoIter = Indices<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = Indices {
            shape: self.0.shape.clone(),
            front: self.0.front,
            back: mid,
            front_cached: self.0.front_cached,
            back_cached: None,
        };
        let right = Indices {
            shape: self.0.shape,
            front: mid,
            back: self.0.back,
            front_cached: None,
            back_cached: self.0.back_cached,
        };

        (Self(left), Self(right))
    }
}
