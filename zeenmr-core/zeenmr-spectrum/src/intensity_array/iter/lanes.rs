use crate::dimension::{DimIndex, Dimension};
use crate::intensity_array::iter::SplitAt;
use crate::intensity_array::{DimOrder, Lane, LaneGeometry, LaneMut, Layout};
use std::iter::FusedIterator;
use std::marker::PhantomData;

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
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
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

impl<D> ExactSizeIterator for LaneGeometries<D> where D: Dimension<Elem = usize> {}

impl<D> FusedIterator for LaneGeometries<D> where D: Dimension<Elem = usize> {}

impl<D> LaneGeometries<D>
where
    D: Dimension<Elem = usize>,
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

// SAFETY: setting left.back and right.front to `self.front + index` achieves
// the disjoint split required by `SplitAt`.
unsafe impl<D> SplitAt for LaneGeometries<D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.front + index;
        let left = Self {
            back: mid,
            ..self.clone()
        };
        let right = Self { front: mid, ..self };

        (left, right)
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
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next_back()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { Lane::from_raw(self.base, geometry) })
    }
}

impl<'s, T, D> ExactSizeIterator for Lanes<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for Lanes<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for Lanes<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.geometries.split_at(index);

        (
            Self {
                base: self.base,
                geometries: left,
                lifetime: PhantomData,
            },
            Self {
                base: self.base,
                geometries: right,
                lifetime: PhantomData,
            },
        )
    }
}

impl<'s, T, D> Lanes<'s, T, D>
where
    D: Dimension<Elem = usize>,
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
    /// Lifetime marker for the mutable reference.
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
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next_back()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { LaneMut::from_raw(self.base, geometry) })
    }
}

impl<'s, T, D> ExactSizeIterator for LanesMut<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for LanesMut<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> LanesMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
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

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for LanesMut<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.geometries.split_at(index);

        (
            Self {
                base: self.base,
                geometries: left,
                lifetime: PhantomData,
            },
            Self {
                base: self.base,
                geometries: right,
                lifetime: PhantomData,
            },
        )
    }
}
