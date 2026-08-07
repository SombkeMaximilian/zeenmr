//! Array iterators.

use crate::intensity_array::{
    ArrayIndex, DimIndex, DimOrder, Dimension, Lane, LaneGeometry, Layout, Shape,
};
use std::iter::FusedIterator;
use std::marker::PhantomData;

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

/// Iterator over the lanes of a layout along one dimension.
#[derive(Clone, Debug)]
pub struct LaneGeometries<D> {
    /// Underlying layout.
    layout: Layout<D>,
    /// Order of the dimensions.
    order: DimOrder<D>,
    /// Dimension the lanes run along.
    dim: DimIndex,
    /// Next lane number from the front.
    front: usize,
    /// Next lane number from the back.
    back: usize,
}

impl<D> Iterator for LaneGeometries<D>
where
    D: Dimension,
{
    type Item = LaneGeometry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = self
                .layout
                .lane_unvalidated(self.dim, self.front, &self.order);
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

impl<D> DoubleEndedIterator for LaneGeometries<D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;

            Some(
                self.layout
                    .lane_unvalidated(self.dim, self.back, &self.order),
            )
        } else {
            None
        }
    }
}

impl<D> ExactSizeIterator for LaneGeometries<D> where D: Dimension {}

impl<D> FusedIterator for LaneGeometries<D> where D: Dimension {}

impl<D> LaneGeometries<D>
where
    D: Dimension,
{
    /// Creates an iterator over the lanes of `layout` along `dim`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `layout`.
    ///
    /// Prefer the `lanes_*` methods on [`Layout`].
    pub fn new(layout: Layout<D>, dim: DimIndex, order: DimOrder<D>) -> Option<Self> {
        if order.rank() != layout.rank() {
            return None;
        }

        let back = layout.lane_count(dim)?;

        Some(Self {
            layout,
            dim,
            order,
            front: 0,
            back,
        })
    }
}

/// Parallel iterator over the lanes of a layout along one dimension.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLaneGeometries<D>(LaneGeometries<D>);

#[cfg(feature = "rayon")]
impl<D> ParallelIterator for ParLaneGeometries<D>
where
    D: Dimension,
{
    type Item = LaneGeometry;

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
impl<D> IndexedParallelIterator for ParLaneGeometries<D>
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
        callback.callback(LaneGeometriesProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<D> ParLaneGeometries<D>
where
    D: Dimension,
{
    /// Creates a parallel iterator over the lanes of `layout` along `dim`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `layout`.
    ///
    /// Prefer the `par_lanes_*` methods on [`Layout`].
    pub fn new(layout: Layout<D>, dim: DimIndex, order: DimOrder<D>) -> Option<Self> {
        Some(Self(LaneGeometries::new(layout, dim, order)?))
    }
}

/// Producer for [`ParLaneGeometries`].
#[cfg(feature = "rayon")]
struct LaneGeometriesProducer<D>(LaneGeometries<D>);

#[cfg(feature = "rayon")]
impl<D> Producer for LaneGeometriesProducer<D>
where
    D: Dimension,
{
    type Item = LaneGeometry;
    type IntoIter = LaneGeometries<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = LaneGeometries {
            back: mid,
            ..self.0.clone()
        };
        let right = LaneGeometries {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}

/// Iterator over the buffer offsets of a lane along one dimension.
#[derive(Clone, Debug)]
pub struct LaneOffsets {
    /// Geometry of the lane.
    geometry: LaneGeometry,
    /// Next lane offset from the front.
    front: usize,
    /// Next lane offset from the back.
    back: usize,
}

impl Iterator for LaneOffsets {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = self.geometry.offset_of_unvalidated(self.front);
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

impl DoubleEndedIterator for LaneOffsets {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;

            Some(self.geometry.offset_of_unvalidated(self.back))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for LaneOffsets {}

impl FusedIterator for LaneOffsets {}

impl LaneOffsets {
    /// Creates an iterator over the offsets of a lane.
    pub fn new(geometry: LaneGeometry) -> Self {
        Self {
            geometry,
            front: 0,
            back: geometry.len(),
        }
    }
}

/// Parallel iterator over the buffer offsets of a lane along one dimension.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLaneOffsets(LaneOffsets);

#[cfg(feature = "rayon")]
impl ParallelIterator for ParLaneOffsets {
    type Item = usize;

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
impl IndexedParallelIterator for ParLaneOffsets {
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
        callback.callback(LaneOffsetsProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl ParLaneOffsets {
    /// Creates a parallel iterator over the offsets of a lane.
    pub fn new(geometry: LaneGeometry) -> Self {
        Self(LaneOffsets::new(geometry))
    }
}

/// Producer for [`ParLaneOffsets`]
#[cfg(feature = "rayon")]
struct LaneOffsetsProducer(LaneOffsets);

#[cfg(feature = "rayon")]
impl Producer for LaneOffsetsProducer {
    type Item = usize;
    type IntoIter = LaneOffsets;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = LaneOffsets {
            geometry: self.0.geometry,
            front: self.0.front,
            back: mid,
        };
        let right = LaneOffsets {
            geometry: self.0.geometry,
            front: mid,
            back: self.0.back,
        };

        (LaneOffsetsProducer(left), LaneOffsetsProducer(right))
    }
}

/// Strided iterator over the elements of a buffer.
///
/// This is the slow path of lane iteration.
#[derive(Clone, Debug)]
pub struct LaneElemStrided<'s, T> {
    /// Reference to the underlying buffer.
    base: &'s [T],
    /// Offsets to visit.
    ///
    /// # Safety
    ///
    /// All offsets returned by this iterator must be within bounds.
    offsets: LaneOffsets,
}

impl<'s, T> Iterator for LaneElemStrided<'s, T> {
    type Item = &'s T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next()?;

        // SAFETY: offsets are all guaranteed to be within bounds by
        // construction of this type
        Some(unsafe { self.base.get_unchecked(next) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<'s, T> DoubleEndedIterator for LaneElemStrided<'s, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next_back()?;

        // SAFETY: offsets are all guaranteed to be within bounds by
        // construction of this type
        Some(unsafe { self.base.get_unchecked(next) })
    }
}

impl<'s, T> ExactSizeIterator for LaneElemStrided<'s, T> {}

impl<'s, T> FusedIterator for LaneElemStrided<'s, T> {}

impl<'s, T> LaneElemStrided<'s, T> {
    /// Creates a strided iterator over the elements of `base` according to
    /// `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds
    /// or overflows.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        geometry.fits_within(base.len()).then(|| Self {
            base,
            offsets: LaneOffsets::new(geometry),
        })
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.offsets.geometry.stride()
    }
}

/// Parallel, strided iterator over the elements of a buffer.
///
/// This is the slow path of lane iteration.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLaneElemStrided<'s, T>(LaneElemStrided<'s, T>);

#[cfg(feature = "rayon")]
impl<'s, T> ParallelIterator for ParLaneElemStrided<'s, T>
where
    T: Sync,
{
    type Item = &'s T;

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
impl<'s, T> IndexedParallelIterator for ParLaneElemStrided<'s, T>
where
    T: Sync,
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
        callback.callback(LaneElemStridedProducer {
            base: self.0.base,
            producer: LaneOffsetsProducer(self.0.offsets),
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> ParLaneElemStrided<'s, T> {
    /// Creates a parallel, strided iterator over the elements of `base`
    /// according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds
    /// or overflows.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        Some(Self(LaneElemStrided::new(base, geometry)?))
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.0.stride()
    }
}

/// Producer for [`ParLaneElemStrided`].
#[cfg(feature = "rayon")]
struct LaneElemStridedProducer<'s, T> {
    /// Reference to the underlying buffer.
    base: &'s [T],
    /// Producer for [`ParLaneOffsets`].
    producer: LaneOffsetsProducer,
}

#[cfg(feature = "rayon")]
impl<'s, T> Producer for LaneElemStridedProducer<'s, T>
where
    T: Sync,
{
    type Item = &'s T;
    type IntoIter = LaneElemStrided<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        LaneElemStrided {
            base: self.base,
            offsets: self.producer.into_iter(),
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        (
            Self {
                base: self.base,
                producer: left,
            },
            Self {
                base: self.base,
                producer: right,
            },
        )
    }
}

/// Iterator over contiguous elements of a buffer.
///
/// This is the fast path of lane iteration.
pub type LaneElemContiguous<'s, T> = std::slice::Iter<'s, T>;

/// Parallel iterator over contiguous elements of a buffer.
///
/// This is the fast path of lane iteration.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguous<'s, T> = rayon::slice::Iter<'s, T>;

/// Iterators over the elements of a lane view.
pub type LaneElem<'s, T> = StridedIterKind<LaneElemContiguous<'s, T>, LaneElemStrided<'s, T>>;

/// Parallel iterators over the elements of a lane view.
#[cfg(feature = "rayon")]
pub type ParLaneElem<'s, T> =
    StridedIterKind<ParLaneElemContiguous<'s, T>, ParLaneElemStrided<'s, T>>;

/// Strided iterator over mutable references of the elements of a buffer.
///
/// This is the slow path of lane iteration.
#[derive(Debug)]
pub struct LaneElemStridedMut<'s, T> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Offsets to visit.
    ///
    /// # Safety
    ///
    /// All offsets returned by this iterator must be within bounds of the
    /// allocation `base` points into, and the geometry must be injective, so
    /// no offset is yielded twice.
    offsets: LaneOffsets,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    ///
    /// # Safety
    ///
    /// This allows the borrow checker to ensure the pointer is never dangling.
    lifetime: PhantomData<&'s mut T>,
}

unsafe impl<T> Send for LaneElemStridedMut<'_, T> where T: Send {}

unsafe impl<T> Sync for LaneElemStridedMut<'_, T> where T: Sync {}

impl<'s, T> Iterator for LaneElemStridedMut<'s, T> {
    type Item = &'s mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next()?;

        // SAFETY: the offset is in bounds by construction, and injectivity
        // guarantees it is never yielded twice, so the returned reference is
        // unique.
        Some(unsafe { &mut *self.base.add(next) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<'s, T> DoubleEndedIterator for LaneElemStridedMut<'s, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next_back()?;

        // SAFETY: the offset is in bounds by construction, and injectivity
        // guarantees it is never yielded twice, so the returned reference is
        // unique.
        Some(unsafe { &mut *self.base.add(next) })
    }
}

impl<'s, T> ExactSizeIterator for LaneElemStridedMut<'s, T> {}

impl<'s, T> FusedIterator for LaneElemStridedMut<'s, T> {}

impl<'s, T> LaneElemStridedMut<'s, T> {
    /// Creates a strided iterator over mutable references of the elements of
    /// `base` according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds or
    /// overflows, or if `geometry` is not injective.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        if !geometry.is_injective() {
            return None;
        }

        geometry.fits_within(base.len()).then(|| Self {
            base: base.as_mut_ptr(),
            offsets: LaneOffsets::new(geometry),
            lifetime: PhantomData,
        })
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.offsets.geometry.stride()
    }
}

/// Parallel, strided iterator over mutable references of the elements of a
/// buffer.
///
/// This is the slow path of lane iteration.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct ParLaneElemStridedMut<'s, T>(LaneElemStridedMut<'s, T>);

#[cfg(feature = "rayon")]
impl<'s, T> ParallelIterator for ParLaneElemStridedMut<'s, T>
where
    T: Send,
{
    type Item = &'s mut T;

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
impl<'s, T> IndexedParallelIterator for ParLaneElemStridedMut<'s, T>
where
    T: Send,
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
        callback.callback(LaneElemStridedMutProducer {
            base: self.0.base,
            producer: LaneOffsetsProducer(self.0.offsets),
            lifetime: PhantomData,
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> ParLaneElemStridedMut<'s, T> {
    /// Creates a parallel, strided iterator over mutable references of the
    /// elements of `base` according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds or
    /// overflows, or if `geometry` is not injective.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        Some(Self(LaneElemStridedMut::new(base, geometry)?))
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.0.stride()
    }
}

/// Producer for [`ParLaneElemStridedMut`].
#[cfg(feature = "rayon")]
struct LaneElemStridedMutProducer<'s, T> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Producer for [`ParLaneOffsets`].
    producer: LaneOffsetsProducer,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    lifetime: PhantomData<&'s mut T>,
}

#[cfg(feature = "rayon")]
unsafe impl<'s, T> Send for LaneElemStridedMutProducer<'s, T> where T: Send {}

#[cfg(feature = "rayon")]
unsafe impl<'s, T> Sync for LaneElemStridedMutProducer<'s, T> where T: Sync {}

#[cfg(feature = "rayon")]
impl<'s, T> Producer for LaneElemStridedMutProducer<'s, T>
where
    T: Send,
{
    type Item = &'s mut T;
    type IntoIter = LaneElemStridedMut<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        LaneElemStridedMut {
            base: self.base,
            offsets: self.producer.into_iter(),
            lifetime: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        // SAFETY: `LaneElemStridedMut` guarantees that any reference returned
        // from it is unique, so splitting it in two without overlap can also
        // only produce unique mutable references.
        (
            Self {
                base: self.base,
                producer: left,
                lifetime: PhantomData,
            },
            Self {
                base: self.base,
                producer: right,
                lifetime: PhantomData,
            },
        )
    }
}

/// Iterator over mutable references of contiguous elements of a buffer.
///
/// This is the fast path of lane iteration.
pub type LaneElemContiguousMut<'s, T> = std::slice::IterMut<'s, T>;

/// Parallel iterator over mutable references of contiguous elements of a
/// buffer.
///
/// This is the fast path of lane iteration.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguousMut<'s, T> = rayon::slice::IterMut<'s, T>;

/// Iterator over the elements of a mutable lane view.
pub type LaneElemMut<'s, T> =
    StridedIterKind<LaneElemContiguousMut<'s, T>, LaneElemStridedMut<'s, T>>;

/// Parallel iterator over the elements of a mutable lane view.
#[cfg(feature = "rayon")]
pub type ParLaneElemMut<'s, T> =
    StridedIterKind<ParLaneElemContiguousMut<'s, T>, ParLaneElemStridedMut<'s, T>>;

/// Sum type containing iterators over strided elements.
///
/// Splitting iteration with strides into the `stride = 1` and `stride = n > 1`
/// case allows for aggressive optimizations on the contiguous, fast path, while
/// also allowing for mitigation of the downsides in the cases with larger
/// strides.
#[derive(Clone, Debug)]
pub enum StridedIterKind<S1, SN> {
    /// Contiguous, fast path.
    Contiguous(S1),
    /// Strided, slow path.
    Strided(SN),
}

/// Delegates the implementation to `S1` and `SN` without having to write
/// infinitely many match statements.
macro_rules! delegate {
    ($self:ident, $method:ident($($arg:expr),*)) => {
        match $self {
            Self::Contiguous(iter) => iter.$method($($arg),*),
            Self::Strided(iter) => iter.$method($($arg),*),
        }
    };
}

// for the future: the `try` methods currently require the unstable trait `Try`
// to be named. we can delegate those methods once it becomes stable.
impl<S1, SN> Iterator for StridedIterKind<S1, SN>
where
    S1: Iterator,
    SN: Iterator<Item = S1::Item>,
{
    type Item = S1::Item;

    fn next(&mut self) -> Option<Self::Item> {
        delegate!(self, next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        delegate!(self, size_hint())
    }

    fn count(self) -> usize {
        delegate!(self, count())
    }

    fn last(self) -> Option<Self::Item> {
        delegate!(self, last())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        delegate!(self, nth(n))
    }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        delegate!(self, for_each(f))
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B {
        delegate!(self, collect())
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        delegate!(self, fold(init, f))
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        delegate!(self, all(f))
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        delegate!(self, any(f))
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        delegate!(self, find(predicate))
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        delegate!(self, find_map(f))
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        delegate!(self, position(predicate))
    }
}

impl<S1, SN> DoubleEndedIterator for StridedIterKind<S1, SN>
where
    S1: DoubleEndedIterator,
    SN: DoubleEndedIterator<Item = S1::Item>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        delegate!(self, next_back())
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        delegate!(self, nth_back(n))
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        delegate!(self, rfold(init, f))
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        delegate!(self, rfind(predicate))
    }
}

impl<S1, SN> ExactSizeIterator for StridedIterKind<S1, SN>
where
    S1: ExactSizeIterator,
    SN: ExactSizeIterator<Item = S1::Item>,
{
}

impl<S1, SN> FusedIterator for StridedIterKind<S1, SN>
where
    S1: FusedIterator,
    SN: FusedIterator<Item = S1::Item>,
{
}

#[cfg(feature = "rayon")]
impl<S1, SN> ParallelIterator for StridedIterKind<S1, SN>
where
    S1: ParallelIterator,
    SN: ParallelIterator<Item = S1::Item>,
{
    type Item = S1::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.drive_unindexed(consumer),
            Self::Strided(iter) => iter.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            Self::Contiguous(iter) => iter.opt_len(),
            Self::Strided(iter) => iter.opt_len(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<S1, SN> IndexedParallelIterator for StridedIterKind<S1, SN>
where
    S1: IndexedParallelIterator,
    SN: IndexedParallelIterator<Item = S1::Item>,
{
    fn len(&self) -> usize {
        match self {
            Self::Contiguous(iter) => iter.len(),
            Self::Strided(iter) => iter.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.drive(consumer),
            Self::Strided(iter) => iter.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.with_producer(callback),
            Self::Strided(iter) => iter.with_producer(callback),
        }
    }
}

impl<S1, SN> StridedIterKind<S1, SN> {
    /// Returns the contained contiguous iterator, or `None` if `self` is
    /// `Strided`.
    pub fn try_into_contiguous(self) -> Option<S1> {
        match self {
            StridedIterKind::Contiguous(iter) => Some(iter),
            StridedIterKind::Strided(_) => None,
        }
    }

    /// Returns the contained strided iterator, or `None` if `self` is
    /// `Contiguous`.
    pub fn try_into_strided(self) -> Option<SN> {
        match self {
            StridedIterKind::Contiguous(_) => None,
            StridedIterKind::Strided(iter) => Some(iter),
        }
    }
}

/// Iterator over the lanes of an array along one dimension.
#[derive(Clone, Debug)]
pub struct Lanes<'s, T, D> {
    /// Reference to the underlying buffer.
    base: &'s [T],
    /// Geometries of the lanes.
    geometries: LaneGeometries<D>,
}

impl<'s, T, D> Iterator for Lanes<'s, T, D>
where
    D: Dimension,
{
    type Item = Lane<'s, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next()?;

        Some(self.lane(geometry))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.geometries.size_hint()
    }
}

impl<'s, T, D> DoubleEndedIterator for Lanes<'s, T, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next_back()?;

        Some(self.lane(geometry))
    }
}

impl<'s, T, D> ExactSizeIterator for Lanes<'s, T, D> where D: Dimension {}

impl<'s, T, D> FusedIterator for Lanes<'s, T, D> where D: Dimension {}

impl<'s, T, D> Lanes<'s, T, D>
where
    D: Dimension,
{
    /// Creates an iterator over the lanes of `layout` along `dim` within
    /// `base`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, if `order` has a different
    /// rank than `layout`, or if `layout` can address an offset past the end
    /// of `base`.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(
        base: &'s [T],
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        if layout.max_offset() >= base.len() {
            return None;
        }

        Some(Self {
            base,
            geometries: LaneGeometries::new(layout, dim, order)?,
        })
    }

    /// Returns the lane for `geometry`.
    ///
    /// # Panics
    ///
    /// Panics in the same scenarios as [`Lane::new`], which never happens if
    /// `geometry` is from `self.geometries` due to the check in [`Lanes::new`].
    fn lane(&self, geometry: LaneGeometry) -> Lane<'s, T> {
        Lane::new(self.base, geometry).expect("layout bounds every lane offset by `max_offset`")
    }
}

/// Parallel iterator over the lanes of an array along one dimension.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLanes<'s, T, D>(Lanes<'s, T, D>);

#[cfg(feature = "rayon")]
impl<'s, T, D> ParallelIterator for ParLanes<'s, T, D>
where
    T: Sync,
    D: Dimension,
{
    type Item = Lane<'s, T>;

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
impl<'s, T, D> IndexedParallelIterator for ParLanes<'s, T, D>
where
    T: Sync,
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
        callback.callback(LanesProducer {
            base: self.0.base,
            producer: LaneGeometriesProducer(self.0.geometries),
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T, D> ParLanes<'s, T, D>
where
    D: Dimension,
{
    /// Creates a parallel iterator over the lanes of `layout` along `dim`
    /// within `base`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, if `order` has a different
    /// rank than `layout`, or if `layout` can address an offset past the end
    /// of `base`.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(
        base: &'s [T],
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        Some(Self(Lanes::new(base, layout, dim, order)?))
    }
}

/// Producer for [`ParLanes`].
#[cfg(feature = "rayon")]
struct LanesProducer<'s, T, D> {
    /// Entire storage the lanes walk.
    base: &'s [T],
    /// Producer for the lane geometries.
    producer: LaneGeometriesProducer<D>,
}

#[cfg(feature = "rayon")]
impl<'s, T, D> Producer for LanesProducer<'s, T, D>
where
    T: Sync,
    D: Dimension,
{
    type Item = Lane<'s, T>;
    type IntoIter = Lanes<'s, T, D>;

    fn into_iter(self) -> Self::IntoIter {
        Lanes {
            base: self.base,
            geometries: self.producer.into_iter(),
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        (
            Self {
                base: self.base,
                producer: left,
            },
            Self {
                base: self.base,
                producer: right,
            },
        )
    }
}
