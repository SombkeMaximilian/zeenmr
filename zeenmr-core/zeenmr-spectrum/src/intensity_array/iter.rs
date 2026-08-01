//! Array iterators.

use crate::intensity_array::{ArrayIndex, DimIndex, Dimension, Lane, Layout, Shape};
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Iterator over the multidimensional indices of a shape.
///
/// Yields indices in row-major order: the last dimension varies fastest.
#[derive(Clone, Debug)]
pub struct IndicesRowMajor<D> {
    /// Underlying shape.
    shape: Shape<D>,
    /// Next linear index from the front.
    front: usize,
    /// Next linear index from the back.
    back: usize,
}

impl<D> Iterator for IndicesRowMajor<D>
where
    D: Dimension,
{
    type Item = ArrayIndex<D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = ArrayIndex::linear_in_shape(self.front, &self.shape);
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

impl<D> DoubleEndedIterator for IndicesRowMajor<D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;

            Some(ArrayIndex::linear_in_shape(self.back, &self.shape))
        } else {
            None
        }
    }
}

impl<D> ExactSizeIterator for IndicesRowMajor<D> where D: Dimension {}

impl<D> FusedIterator for IndicesRowMajor<D> where D: Dimension {}

impl<D> IndicesRowMajor<D>
where
    D: Dimension,
{
    /// Creates a row-major index iterator.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    ///
    /// Prefer the methods on the concrete dimension types.
    pub fn new(shape: Shape<D>) -> Option<Self> {
        let back = shape.product_checked()?;

        Some(Self {
            shape,
            front: 0,
            back,
        })
    }
}

/// Parallel iterator over the multidimensional indices of a shape.
///
/// Yields indices in row-major order: the last dimension varies fastest.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParIndicesRowMajor<D>(IndicesRowMajor<D>);

#[cfg(feature = "rayon")]
impl<D> ParallelIterator for ParIndicesRowMajor<D>
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
impl<D> IndexedParallelIterator for ParIndicesRowMajor<D>
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
        callback.callback(IndicesRowMajorProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<D> ParIndicesRowMajor<D>
where
    D: Dimension,
{
    /// Creates a parallel row-major index iterator.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    ///
    /// Prefer the methods on the concrete dimension types.
    pub fn new(shape: Shape<D>) -> Option<Self> {
        Some(Self(IndicesRowMajor::new(shape)?))
    }
}

/// Producer for [`ParIndicesRowMajor`].
#[cfg(feature = "rayon")]
struct IndicesRowMajorProducer<D>(IndicesRowMajor<D>);

#[cfg(feature = "rayon")]
impl<D> Producer for IndicesRowMajorProducer<D>
where
    D: Dimension,
{
    type Item = ArrayIndex<D>;
    type IntoIter = IndicesRowMajor<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = IndicesRowMajor {
            back: mid,
            ..self.0.clone()
        };
        let right = IndicesRowMajor {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}

/// Iterator over the lanes of a layout along one dimension.
///
/// Yields lanes in row-major order over the dimensions other than the lane
/// dimension.
#[derive(Clone, Debug)]
pub struct LanesRowMajor<D> {
    /// Underlying layout.
    layout: Layout<D>,
    /// Dimension the lanes run along.
    dim: DimIndex,
    /// Next lane number from the front.
    front: usize,
    /// Next lane number from the back.
    back: usize,
}

impl<D> Iterator for LanesRowMajor<D>
where
    D: Dimension,
{
    type Item = Lane;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = self.layout.lane_row_major_unvalidated(self.dim, self.front);
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

impl<D> DoubleEndedIterator for LanesRowMajor<D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;

            Some(self.layout.lane_row_major_unvalidated(self.dim, self.back))
        } else {
            None
        }
    }
}

impl<D> ExactSizeIterator for LanesRowMajor<D> where D: Dimension {}

impl<D> FusedIterator for LanesRowMajor<D> where D: Dimension {}

impl<D> LanesRowMajor<D>
where
    D: Dimension,
{
    /// Creates a row-major iterator over the lanes of `layout` along `dim`.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn new(layout: Layout<D>, dim: DimIndex) -> Option<Self> {
        let back = layout.lane_count(dim)?;

        Some(Self { layout, dim, front: 0, back })
    }
}

/// Parallel iterator over the lanes of a layout along one dimension.
///
/// Yields lanes in row-major order over the dimensions other than the lane
/// dimension.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLanesRowMajor<D>(LanesRowMajor<D>);

#[cfg(feature = "rayon")]
impl<D> ParallelIterator for ParLanesRowMajor<D>
where
    D: Dimension,
{
    type Item = Lane;

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
impl<D> IndexedParallelIterator for ParLanesRowMajor<D>
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
        callback.callback(LanesRowMajorProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<D> ParLanesRowMajor<D>
where
    D: Dimension,
{
    /// Creates a row-major, parallel iterator over the lanes of `layout` along
    /// `dim`.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn new(layout: Layout<D>, dim: DimIndex) -> Option<Self> {
        Some(Self(LanesRowMajor::new(layout, dim)?))
    }
}

/// Producer for [`ParLanesRowMajor`].
#[cfg(feature = "rayon")]
struct LanesRowMajorProducer<D>(LanesRowMajor<D>);

#[cfg(feature = "rayon")]
impl<D> Producer for LanesRowMajorProducer<D>
where
    D: Dimension,
{
    type Item = Lane;
    type IntoIter = LanesRowMajor<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = LanesRowMajor {
            back: mid,
            ..self.0.clone()
        };
        let right = LanesRowMajor {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}
