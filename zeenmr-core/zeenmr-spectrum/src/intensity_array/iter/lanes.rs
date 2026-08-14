use crate::intensity_array::{DimIndex, DimOrder, Dimension, Lane, LaneGeometry, LaneMut, Layout};
use std::iter::FusedIterator;
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

/// Iterator over the lane views of an array along one dimension.
#[derive(Debug)]
pub struct Lanes<'s, T, D> {
    /// Base pointer of the storage.
    base: *const T,
    /// Geometries of the lanes.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address offsets within
    /// bounds of `base`.
    geometries: LaneGeometries<D>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s T>,
}

impl<T, D> Clone for Lanes<'_, T, D>
where
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self {
            base: self.base,
            geometries: self.geometries.clone(),
            lifetime: self.lifetime,
        }
    }
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync and `D`
// is Send.
unsafe impl<T, D> Send for Lanes<'_, T, D>
where
    T: Sync,
    D: Send,
{
}

// SAFETY: grants shared access to `T`, so it is Sync when `T` is Sync and `D`
// is Sync.
unsafe impl<T, D> Sync for Lanes<'_, T, D>
where
    T: Sync,
    D: Sync,
{
}

impl<'s, T, D> Iterator for Lanes<'s, T, D>
where
    D: Dimension,
{
    type Item = Lane<'s, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { Lane::from_raw(self.base, geometry) })
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

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { Lane::from_raw(self.base, geometry) })
    }
}

impl<'s, T, D> ExactSizeIterator for Lanes<'s, T, D> where D: Dimension {}

impl<'s, T, D> FusedIterator for Lanes<'s, T, D> where D: Dimension {}

impl<'s, T, D> Lanes<'s, T, D>
where
    D: Dimension,
{
    /// Creates an iterator over the lane views of `layout` along `dim` within
    /// `base`.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
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
            base: base.as_ptr(),
            geometries: LaneGeometries::new(layout, dim, order)?,
            lifetime: PhantomData,
        })
    }
}

/// Parallel iterator over the lane views of an array along one dimension.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct ParLanes<'s, T, D>(Lanes<'s, T, D>);

#[cfg(feature = "rayon")]
impl<'s, T, D> Clone for ParLanes<'s, T, D>
where
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

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
            lifetime: PhantomData,
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T, D> ParLanes<'s, T, D>
where
    D: Dimension,
{
    /// Creates a parallel iterator over the lane views of `layout` along `dim`
    /// within `base`.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
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
    /// Base pointer of the storage.
    base: *const T,
    /// Producer for the [`ParLaneGeometries`].
    producer: LaneGeometriesProducer<D>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s T>,
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync and `D`
// is Sync.
#[cfg(feature = "rayon")]
unsafe impl<T, D> Send for LanesProducer<'_, T, D>
where
    T: Sync,
    D: Sync,
{
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
            lifetime: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

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

/// Iterator over mutable lane views of an array along one dimension.
#[derive(Debug)]
pub struct LanesMut<'s, T, D> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Geometries of the lanes.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address offsets within
    /// bounds of `base`, and they must collectively be injective, s.t. no two
    /// lanes ever address the same offset.
    geometries: LaneGeometries<D>,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    lifetime: PhantomData<&'s mut T>,
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send and `D`
// is Send.
unsafe impl<T, D> Send for LanesMut<'_, T, D> where T: Send {}

// SAFETY: grants unique access to `T`, so it is Sync when `T` is Sync and `D`
// is Sync.
unsafe impl<T, D> Sync for LanesMut<'_, T, D> where T: Sync {}

impl<'s, T, D> Iterator for LanesMut<'s, T, D>
where
    D: Dimension,
{
    type Item = LaneMut<'s, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { LaneMut::from_raw(self.base, geometry) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.geometries.size_hint()
    }
}

impl<'s, T, D> DoubleEndedIterator for LanesMut<'s, T, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next_back()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { LaneMut::from_raw(self.base, geometry) })
    }
}

impl<'s, T, D> ExactSizeIterator for LanesMut<'s, T, D> where D: Dimension {}

impl<'s, T, D> FusedIterator for LanesMut<'s, T, D> where D: Dimension {}

impl<'s, T, D> LanesMut<'s, T, D>
where
    D: Dimension,
{
    /// Creates an iterator over the mutable lane views of `layout` along `dim`
    /// within `base`.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, if `order` has a different
    /// rank than `layout`, or if `layout` can address an offset past the end
    /// of `base`, or if `layout` is self-overlapping.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(
        base: &'s mut [T],
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        if layout.max_offset() >= base.len() || !layout.is_non_overlapping() {
            return None;
        }

        Some(Self {
            base: base.as_mut_ptr(),
            geometries: LaneGeometries::new(layout, dim, order)?,
            lifetime: PhantomData,
        })
    }
}

/// Parallel iterator over the mutable lane views of an array along one
/// dimension.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct ParLanesMut<'s, T, D>(LanesMut<'s, T, D>);

#[cfg(feature = "rayon")]
impl<'s, T, D> ParallelIterator for ParLanesMut<'s, T, D>
where
    T: Send,
    D: Dimension,
{
    type Item = LaneMut<'s, T>;

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
impl<'s, T, D> IndexedParallelIterator for ParLanesMut<'s, T, D>
where
    T: Send,
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
        callback.callback(LanesMutProducer {
            base: self.0.base,
            producer: LaneGeometriesProducer(self.0.geometries),
            lifetime: PhantomData,
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T, D> ParLanesMut<'s, T, D>
where
    D: Dimension,
{
    /// Creates a parallel iterator over the mutable lane views of `layout`
    /// along `dim` within `base`.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, if `order` has a different
    /// rank than `layout`, or if `layout` can address an offset past the end
    /// of `base`, or if `layout` is self-overlapping.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(
        base: &'s mut [T],
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        Some(Self(LanesMut::new(base, layout, dim, order)?))
    }
}

/// Producer for [`ParLanesMut`].
#[cfg(feature = "rayon")]
struct LanesMutProducer<'s, T, D> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Producer for the [`ParLaneGeometries`].
    producer: LaneGeometriesProducer<D>,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    lifetime: PhantomData<&'s mut T>,
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send.
#[cfg(feature = "rayon")]
unsafe impl<T, D> Send for LanesMutProducer<'_, T, D> where T: Send {}

#[cfg(feature = "rayon")]
impl<'s, T, D> Producer for LanesMutProducer<'s, T, D>
where
    T: Send,
    D: Dimension,
{
    type Item = LaneMut<'s, T>;
    type IntoIter = LanesMut<'s, T, D>;

    fn into_iter(self) -> Self::IntoIter {
        LanesMut {
            base: self.base,
            geometries: self.producer.into_iter(),
            lifetime: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        // SAFETY: `LanesMut` guarantees that any lanes views returned from it
        // are non-overlapping, so splitting it in two without overlap can also
        // only produce non-overlapping lane views.
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
