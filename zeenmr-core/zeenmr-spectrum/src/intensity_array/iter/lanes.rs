use crate::dimension::{DimIndex, Dimension};
use crate::intensity_array::iter::SplitAt;
use crate::intensity_array::storage::{RawAccess, RawAccessMut};
use crate::intensity_array::{DimOrder, Lane, LaneGeometry, LaneMut, Layout};
use std::iter::FusedIterator;

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
    /// Access pointer of the storage
    access: RawAccess<'s, T>,
    /// Geometries of the lanes.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address valid offsets into
    /// the allocation `access` points to.
    geometries: LaneGeometries<D>,
}

impl<T, D> Clone for Lanes<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn clone(&self) -> Self {
        Self {
            access: self.access,
            geometries: self.geometries.clone(),
        }
    }
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
        Some(unsafe { Lane::from_access(self.access, geometry) })
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
        Some(unsafe { Lane::from_access(self.access, geometry) })
    }
}

impl<'s, T, D> ExactSizeIterator for Lanes<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for Lanes<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`, and disjoint lane numbers imply disjoint
// offsets.
unsafe impl<T, D> SplitAt for Lanes<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.geometries.split_at(index);

        (
            Self {
                access: self.access,
                geometries: left,
            },
            Self {
                access: self.access,
                geometries: right,
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
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does.
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
            access: RawAccess::from_slice(base),
            geometries: LaneGeometries::new(layout, dim, order)?,
        })
    }

    /// Creates an iterator over the lane views of `layout` along `dim` within
    /// the allocation `access` points into.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    ///
    /// # Safety
    ///
    /// Every offset layout addresses must be a valid offset into the allocation
    /// `access` points into, and its elements must be borrowed immutably for
    /// `'s`.
    pub(crate) unsafe fn from_access(
        access: RawAccess<'s, T>,
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        Some(Self {
            access,
            geometries: LaneGeometries::new(layout, dim, order)?,
        })
    }
}

/// Iterator over mutable lane views of an array along one dimension.
#[derive(Debug)]
pub struct LanesMut<'s, T, D> {
    /// Access pointer of the storage
    access: RawAccessMut<'s, T>,
    /// Geometries of the lanes.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address valid offsets into
    /// the allocation `access` points to, and they must collectively be
    /// injective, s.t. no two lanes collectively ever address the same offset
    /// more than once.
    geometries: LaneGeometries<D>,
}

impl<'s, T, D> Iterator for LanesMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    type Item = LaneMut<'s, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let geometry = self.geometries.next()?;

        // SAFETY: geometries returned by `self.geometries` are guaranteed to
        // fulfil the requirements.
        Some(unsafe { LaneMut::from_access(self.access, geometry) })
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
        Some(unsafe { LaneMut::from_access(self.access, geometry) })
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
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// and if `layout` is self-overlapping.
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
            access: RawAccessMut::from_slice(base),
            geometries: LaneGeometries::new(layout, dim, order)?,
        })
    }

    /// Creates an iterator over the mutable lane views of `layout` along `dim`
    /// within the allocation `access` points into.
    ///
    /// Lane views are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// and if `layout` is self-overlapping.
    ///
    /// Prefer the `lanes_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    ///
    /// # Safety
    ///
    /// Every offset layout addresses must be a valid offset into the allocation
    /// `access` points into, and its elements must be borrowed mutably for
    /// `'s`.
    ///
    /// Disjointness of the lanes is already established by the layout being
    /// required to be non-overlapping (otherwise `None` is returned).
    pub(crate) unsafe fn from_access(
        access: RawAccessMut<'s, T>,
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        if !layout.is_non_overlapping() {
            return None;
        }

        Some(Self {
            access,
            geometries: LaneGeometries::new(layout, dim, order)?,
        })
    }
}

// SAFETY: see `LaneGeometries`, and disjoint lane numbers imply disjoint
// offsets.
unsafe impl<T, D> SplitAt for LanesMut<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.geometries.split_at(index);

        (
            Self {
                access: self.access,
                geometries: left,
            },
            Self {
                access: self.access,
                geometries: right,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::DynDim;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(LaneGeometries<DynDim<usize>>: Send, Sync);
        assert_impl_all!(Lanes<'_, u8, DynDim<usize>>: Send, Sync);
        assert_impl_all!(LanesMut<'_, u8, DynDim<usize>>: Send, Sync);
    }
}
