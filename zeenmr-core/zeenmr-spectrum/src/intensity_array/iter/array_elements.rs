use crate::dimension::Dimension;
use crate::intensity_array::iter::{
    LaneElemContiguous, LaneElemContiguousMut, LaneElemStrided, LaneElemStridedMut, LaneGeometries,
};
use crate::intensity_array::storage::{RawAccess, RawAccessMut};
use crate::intensity_array::{DimOrder, LaneGeometry, Layout};
use crate::iter::SplitAt;
use std::iter::FusedIterator;

// NOTE: It may seem like a shared trait or macro may make some of the
// repetition disappear here. If a reader can figure out a way to do that AND
// make it readable, you're certainly better at this than I am. I've already
// tried both a few traits and a macro, and it just ended up unreadable, so I'm
// placing the burden of repetition onto the future.

/// Contiguous iterator over the elements of an array.
#[derive(Debug)]
pub struct ArrayElemContiguous<'s, T, D> {
    /// Access pointer of the storage.
    access: RawAccess<'s, T>,
    /// Lane geometries.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address valid offsets into
    /// the allocation `access` points to, and must be contiguous.
    geometries: LaneGeometries<D>,
    /// Element iterator at the front.
    front: LaneElemContiguous<'s, T>,
    /// Element iterator at the back.
    back: LaneElemContiguous<'s, T>,
    /// Element count of each lane.
    lane_len: usize,
}

impl<T, D> Clone for ArrayElemContiguous<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn clone(&self) -> Self {
        Self {
            access: self.access,
            geometries: self.geometries.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            lane_len: self.lane_len,
        }
    }
}

impl<'s, T, D> Iterator for ArrayElemContiguous<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    type Item = &'s T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.front.next() {
            Some(next) => Some(next),
            None if let Some(next_slice) = self.next_slice_front() => {
                self.front = next_slice.iter();

                self.front.next()
            }
            None => self.back.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.front.len() + self.geometries.len() * self.lane_len + self.back.len();

        (remaining, Some(remaining))
    }
}

impl<'s, T, D> DoubleEndedIterator for ArrayElemContiguous<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.back.next_back() {
            Some(next) => Some(next),
            None if let Some(next_slice) = self.next_slice_back() => {
                self.back = next_slice.iter();

                self.back.next_back()
            }
            None => self.front.next_back(),
        }
    }
}

impl<'s, T, D> ExactSizeIterator for ArrayElemContiguous<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for ArrayElemContiguous<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for ArrayElemContiguous<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let front_len = self.front.len();
        let mid_len = self.geometries.len() * self.lane_len;
        let index = index.min(front_len + mid_len + self.back.len());

        if index <= front_len {
            let (left, right) = self.front.as_slice().split_at(index);
            let (empty, geometries) = self.geometries.split_at(0);

            (
                Self {
                    access: self.access,
                    geometries: empty,
                    front: left.iter(),
                    back: [].iter(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries,
                    front: right.iter(),
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else if index <= front_len + mid_len {
            let shifted = index - front_len;
            let (left_geoms, mut right_geoms) = self.geometries.split_at(shifted / self.lane_len);
            let mid_split = shifted % self.lane_len;

            let (left_back, right_front) = if mid_split == 0 {
                ([].iter(), [].iter())
            } else if let Some(geometry) = right_geoms.next() {
                // SAFETY: the type invariant guarantees that every offset
                // addressed by any geometry of `self.geometries` is contiguous
                // and within bounds of the allocation `access` points to.
                let lane = unsafe { Self::make_slice(self.access, geometry) };

                (lane[..mid_split].iter(), lane[mid_split..].iter())
            } else {
                ([].iter(), [].iter())
            };

            (
                Self {
                    access: self.access,
                    geometries: left_geoms,
                    front: self.front,
                    back: left_back,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: right_geoms,
                    front: right_front,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else {
            let shifted = index - front_len - mid_len;
            let (left, right) = self.back.as_slice().split_at(shifted);
            let geometries_len = self.geometries.len();
            let (geometries, empty) = self.geometries.split_at(geometries_len);

            (
                Self {
                    access: self.access,
                    geometries,
                    front: self.front,
                    back: left.iter(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: empty,
                    front: right.iter(),
                    back: [].iter(),
                    lane_len: self.lane_len,
                },
            )
        }
    }
}

impl<'s, T, D> ArrayElemContiguous<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    /// Creates an iterator over the elements addressed by `layout` within
    /// `base`.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - [`Layout::max_offset`] is not less than the number of elements in
    ///   `base`,
    /// - `layout` and `order` have different ranks, or
    /// - elements along the last dimension of `order` are not contiguous in the
    ///   layout.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(base: &'s [T], layout: Layout<D>, order: DimOrder<D>) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank()
            || layout.max_offset() >= base.len()
            || !layout.lanes_are_contiguous(dim)
        {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access: RawAccess::from_slice(base),
            geometries,
            front: [].iter(),
            back: [].iter(),
            lane_len,
        })
    }

    /// Creates an iterator over the elements addressed by `layout` within the
    /// allocation `access` points to.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - `layout` and `order` have different ranks, or
    /// - elements along the last dimension of `order` are not contiguous in the
    ///   layout.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
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
        order: DimOrder<D>,
    ) -> Option<Self> {
        let dim = order.last()?;

        if !layout.lanes_are_contiguous(dim) {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access,
            geometries,
            front: [].iter(),
            back: [].iter(),
            lane_len,
        })
    }

    /// Returns the next slice from the front.
    fn next_slice_front(&mut self) -> Option<&'s [T]> {
        let geometry = self.geometries.next()?;
        // SAFETY: the type invariant guarantees that every offset addressed
        // by any geometry of `self.geometries` is contiguous and within bounds
        // of the allocation `access` points to.
        let slice = unsafe { Self::make_slice(self.access, geometry) };

        Some(slice)
    }

    /// Returns the next slice from the back.
    fn next_slice_back(&mut self) -> Option<&'s [T]> {
        let geometry = self.geometries.next_back()?;
        // SAFETY: see above.
        let slice = unsafe { Self::make_slice(self.access, geometry) };

        Some(slice)
    }

    /// Makes a slice from an access pointer and a lane geometry.
    ///
    /// Note that both the below clauses are always taken care of if `access`
    /// and `geometry` are obtained from the same source array that is
    /// contiguous in the lane dimension.
    ///
    /// # Panics
    ///
    /// Panics if [`LaneGeometry::contiguous_range`] returns `None` for
    /// `geometry`.
    ///
    /// # Safety
    ///
    /// Every offset addressed by `geometry` must be within bounds of the
    /// allocation `access` points to.
    unsafe fn make_slice(access: RawAccess<'s, T>, geometry: LaneGeometry) -> &'s [T] {
        let range = geometry
            .contiguous_range()
            .expect("caller guarantees this");

        // SAFETY: the caller guarantees that every offsets addressed by
        // `geometry` is within bounds of the allocation `access` points to.
        unsafe { std::slice::from_raw_parts(access.as_ptr().add(range.start), range.len()) }
    }
}

/// Contiguous iterator over mutable references to the elements of an array.
#[derive(Debug)]
pub struct ArrayElemContiguousMut<'s, T, D> {
    /// Access pointer of the storage.
    access: RawAccessMut<'s, T>,
    /// Lane geometries.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must
    /// - only address valid offsets into the allocation `access` points to,
    /// - collectively be injective, s.t. no two lanes collectively ever address
    ///   the same offset more than once, and
    /// - be contiguous.
    geometries: LaneGeometries<D>,
    /// Element iterator at the front.
    front: LaneElemContiguousMut<'s, T>,
    /// Element iterator at the back.
    back: LaneElemContiguousMut<'s, T>,
    /// Element count of each lane.
    lane_len: usize,
}

impl<'s, T, D> Iterator for ArrayElemContiguousMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    type Item = &'s mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.front.next() {
            Some(next) => Some(next),
            None if let Some(next_slice) = self.next_mut_slice_front() => {
                self.front = next_slice.iter_mut();

                self.front.next()
            }
            None => self.back.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.front.len() + self.geometries.len() * self.lane_len + self.back.len();

        (remaining, Some(remaining))
    }
}

impl<'s, T, D> DoubleEndedIterator for ArrayElemContiguousMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.back.next_back() {
            Some(next) => Some(next),
            None if let Some(next_slice) = self.next_mut_slice_back() => {
                self.back = next_slice.iter_mut();

                self.back.next_back()
            }
            None => self.front.next_back(),
        }
    }
}

impl<'s, T, D> ExactSizeIterator for ArrayElemContiguousMut<'s, T, D> where
    D: Dimension<Elem = usize>
{
}

impl<'s, T, D> FusedIterator for ArrayElemContiguousMut<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for ArrayElemContiguousMut<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let front_len = self.front.len();
        let mid_len = self.geometries.len() * self.lane_len;
        let index = index.min(front_len + mid_len + self.back.len());

        if index <= front_len {
            let (left, right) = self.front.into_slice().split_at_mut(index);
            let (empty, geometries) = self.geometries.split_at(0);

            (
                Self {
                    access: self.access,
                    geometries: empty,
                    front: left.iter_mut(),
                    back: [].iter_mut(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries,
                    front: right.iter_mut(),
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else if index <= front_len + mid_len {
            let shifted = index - front_len;
            let (left_geoms, mut right_geoms) = self.geometries.split_at(shifted / self.lane_len);
            let mid_split = shifted % self.lane_len;

            let (left_back, right_front) = if mid_split == 0 {
                ([].iter_mut(), [].iter_mut())
            } else if let Some(geometry) = right_geoms.next() {
                // SAFETY: the type invariant guarantees that every offset
                // addressed by any geometry of `self.geometries` is unique,
                // contiguous, and within bounds of the allocation `access`
                // points to.
                let lane = unsafe { Self::make_mut_slice(self.access, geometry) };
                let (left_lane, right_lane) = lane.split_at_mut(mid_split);

                (left_lane.iter_mut(), right_lane.iter_mut())
            } else {
                ([].iter_mut(), [].iter_mut())
            };

            (
                Self {
                    access: self.access,
                    geometries: left_geoms,
                    front: self.front,
                    back: left_back,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: right_geoms,
                    front: right_front,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else {
            let shifted = index - front_len - mid_len;
            let (left, right) = self.back.into_slice().split_at_mut(shifted);
            let geometries_len = self.geometries.len();
            let (geometries, empty) = self.geometries.split_at(geometries_len);

            (
                Self {
                    access: self.access,
                    geometries,
                    front: self.front,
                    back: left.iter_mut(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: empty,
                    front: right.iter_mut(),
                    back: [].iter_mut(),
                    lane_len: self.lane_len,
                },
            )
        }
    }
}

impl<'s, T, D> ArrayElemContiguousMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    /// Creates an iterator over mutable references to elements addressed by
    /// `layout` within `base`.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - [`Layout::max_offset`] is not less than the number of elements in
    ///   `base`,
    /// - the layout is not non-overlapping,
    /// - `layout` and `order` have different ranks, or
    /// - elements along the last dimension of `order` are not contiguous in the
    ///   layout.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(base: &'s mut [T], layout: Layout<D>, order: DimOrder<D>) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank()
            || layout.max_offset() >= base.len()
            || !layout.lanes_are_contiguous(dim)
            || !layout.is_non_overlapping()
        {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access: RawAccessMut::from_slice(base),
            geometries,
            front: [].iter_mut(),
            back: [].iter_mut(),
            lane_len,
        })
    }

    /// Creates an iterator over mutable references to elements addressed by
    /// `layout` within the allocation `access` points to.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - the layout is not non-overlapping,
    /// - `layout` and `order` have different ranks, or
    /// - elements along the last dimension of `order` are not contiguous in the
    ///   layout.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    ///
    /// # Safety
    ///
    /// Every offset layout addresses must be a valid offset into the allocation
    /// `access` points into, and its elements must be borrowed mutably for
    /// `'s`.
    pub(crate) unsafe fn from_access(
        access: RawAccessMut<'s, T>,
        layout: Layout<D>,
        order: DimOrder<D>,
    ) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank()
            || !layout.lanes_are_contiguous(dim)
            || !layout.is_non_overlapping()
        {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access,
            geometries,
            front: [].iter_mut(),
            back: [].iter_mut(),
            lane_len,
        })
    }

    /// Returns the next mutable slice from the front.
    fn next_mut_slice_front(&mut self) -> Option<&'s mut [T]> {
        let geometry = self.geometries.next()?;
        // SAFETY: the type invariant guarantees that every offset addressed
        // by any geometry of `self.geometries` is unique, contiguous, and
        // within bounds of the allocation `access` points to.
        let slice = unsafe { Self::make_mut_slice(self.access, geometry) };

        Some(slice)
    }

    /// Returns the next mutable slice from the back.
    fn next_mut_slice_back(&mut self) -> Option<&'s mut [T]> {
        let geometry = self.geometries.next_back()?;
        // SAFETY: see above.
        let slice = unsafe { Self::make_mut_slice(self.access, geometry) };

        Some(slice)
    }

    /// Makes a mutable slice from an access pointer and a lane geometry.
    ///
    /// Note that both the below clauses are always taken care of if `access`
    /// and `geometry` are obtained from the same source array with a
    /// non-overlapping layout that is contiguous in the lane dimension.
    ///
    /// # Panics
    ///
    /// Panics if [`LaneGeometry::contiguous_range`] returns `None` for
    /// `geometry`.
    ///
    /// # Safety
    ///
    /// Every offset addressed by `geometry` must be unique and within bounds
    /// of the allocation `access` points to.
    unsafe fn make_mut_slice(
        mut access: RawAccessMut<'s, T>,
        geometry: LaneGeometry,
    ) -> &'s mut [T] {
        let range = geometry
            .contiguous_range()
            .expect("caller guarantees this");

        // SAFETY: the caller guarantees that every offsets addressed by
        // `geometry` is unique and within bounds of the allocation `access`
        // points to.
        unsafe { std::slice::from_raw_parts_mut(access.as_mut_ptr().add(range.start), range.len()) }
    }
}

/// Strided iterator over the elements of an array.
#[derive(Debug)]
pub struct ArrayElemStrided<'s, T, D> {
    /// Access pointer of the storage.
    access: RawAccess<'s, T>,
    /// Lane geometries.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address valid offsets into
    /// the allocation `access` points to.
    geometries: LaneGeometries<D>,
    /// Element iterator at the front.
    front: LaneElemStrided<'s, T>,
    /// Element iterator at the back.
    back: LaneElemStrided<'s, T>,
    /// Element count of each lane.
    lane_len: usize,
}

impl<T, D> Clone for ArrayElemStrided<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn clone(&self) -> Self {
        Self {
            access: self.access,
            geometries: self.geometries.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            lane_len: self.lane_len,
        }
    }
}

impl<'s, T, D> Iterator for ArrayElemStrided<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    type Item = &'s T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.front.next() {
            Some(next) => Some(next),
            None if let Some(next_lane) = self.next_lane_front() => {
                self.front = next_lane;

                self.front.next()
            }
            None => self.back.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.front.len() + self.geometries.len() * self.lane_len + self.back.len();

        (remaining, Some(remaining))
    }
}

impl<'s, T, D> DoubleEndedIterator for ArrayElemStrided<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.back.next_back() {
            Some(next) => Some(next),
            None if let Some(next_lane) = self.next_lane_back() => {
                self.back = next_lane;

                self.back.next_back()
            }
            None => self.front.next_back(),
        }
    }
}

impl<'s, T, D> ExactSizeIterator for ArrayElemStrided<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for ArrayElemStrided<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for ArrayElemStrided<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let front_len = self.front.len();
        let mid_len = self.geometries.len() * self.lane_len;
        let index = index.min(front_len + mid_len + self.back.len());

        if index <= front_len {
            let (left, right) = self.front.split_at(index);
            let (empty, geometries) = self.geometries.split_at(0);

            (
                Self {
                    access: self.access,
                    geometries: empty,
                    front: left,
                    back: empty_lane(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries,
                    front: right,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else if index <= front_len + mid_len {
            let shifted = index - front_len;
            let (left_geoms, mut right_geoms) = self.geometries.split_at(shifted / self.lane_len);
            let mid_split = shifted % self.lane_len;

            let (left_back, right_front) = if mid_split == 0 {
                (empty_lane(), empty_lane())
            } else if let Some(geometry) = right_geoms.next() {
                // SAFETY: the type invariant guarantees that every offset
                // addressed by any geometry of `self.geometries` is guaranteed
                // to be within bounds of the allocation `access` points to.
                let lane = unsafe { LaneElemStrided::from_access(self.access, geometry) };

                lane.split_at(mid_split)
            } else {
                (empty_lane(), empty_lane())
            };

            (
                Self {
                    access: self.access,
                    geometries: left_geoms,
                    front: self.front,
                    back: left_back,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: right_geoms,
                    front: right_front,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else {
            let shifted = index - front_len - mid_len;
            let (left, right) = self.back.split_at(shifted);
            let geometries_len = self.geometries.len();
            let (geometries, empty) = self.geometries.split_at(geometries_len);

            (
                Self {
                    access: self.access,
                    geometries,
                    front: self.front,
                    back: left,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: empty,
                    front: right,
                    back: empty_lane(),
                    lane_len: self.lane_len,
                },
            )
        }
    }
}

impl<'s, T, D> ArrayElemStrided<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    /// Creates an iterator over the elements addressed by `layout` within
    /// `base`.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - [`Layout::max_offset`] is not less than the number of elements in
    ///   `base`, or
    /// - `layout` and `order` have different ranks.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(base: &'s [T], layout: Layout<D>, order: DimOrder<D>) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank() || layout.max_offset() >= base.len() {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access: RawAccess::from_slice(base),
            geometries,
            front: empty_lane(),
            back: empty_lane(),
            lane_len,
        })
    }

    /// Creates an iterator over the elements addressed by `layout` within the
    /// allocation `access` points to.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// or if `layout` and `order` have different ranks.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
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
        order: DimOrder<D>,
    ) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank() {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access,
            geometries,
            front: empty_lane(),
            back: empty_lane(),
            lane_len,
        })
    }

    /// Returns the next lane from the front.
    fn next_lane_front(&mut self) -> Option<LaneElemStrided<'s, T>> {
        let geometry = self.geometries.next()?;
        // SAFETY: the type invariant guarantees that every offset addressed
        // by any geometry of `self.geometries` is within bounds of the
        // allocation `access` points to.
        let lane = unsafe { LaneElemStrided::from_access(self.access, geometry) };

        Some(lane)
    }

    /// Returns the next lane from the back.
    fn next_lane_back(&mut self) -> Option<LaneElemStrided<'s, T>> {
        let geometry = self.geometries.next_back()?;
        // SAFETY: see above.
        let lane = unsafe { LaneElemStrided::from_access(self.access, geometry) };

        Some(lane)
    }
}

/// Strided iterator over mutable references to the elements of an array.
#[derive(Debug)]
pub struct ArrayElemStridedMut<'s, T, D> {
    /// Access pointer of the storage.
    access: RawAccessMut<'s, T>,
    /// Lane geometries.
    ///
    /// # Safety
    ///
    /// All lanes returned by this iterator must only address valid offsets into
    /// the allocation `access` points to, and they must collectively be
    /// injective, s.t. no two lanes collectively ever address the same offset
    /// more than once.
    geometries: LaneGeometries<D>,
    /// Element iterator at the front.
    front: LaneElemStridedMut<'s, T>,
    /// Element iterator at the back.
    back: LaneElemStridedMut<'s, T>,
    /// Element count of each lane.
    lane_len: usize,
}

impl<'s, T, D> Iterator for ArrayElemStridedMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    type Item = &'s mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.front.next() {
            Some(next) => Some(next),
            None if let Some(next_lane) = self.next_mut_lane_front() => {
                self.front = next_lane;

                self.front.next()
            }
            None => self.back.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.front.len() + self.geometries.len() * self.lane_len + self.back.len();

        (remaining, Some(remaining))
    }
}

impl<'s, T, D> DoubleEndedIterator for ArrayElemStridedMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.back.next_back() {
            Some(next) => Some(next),
            None if let Some(next_lane) = self.next_mut_lane_back() => {
                self.back = next_lane;

                self.back.next_back()
            }
            None => self.front.next_back(),
        }
    }
}

impl<'s, T, D> ExactSizeIterator for ArrayElemStridedMut<'s, T, D> where D: Dimension<Elem = usize> {}

impl<'s, T, D> FusedIterator for ArrayElemStridedMut<'s, T, D> where D: Dimension<Elem = usize> {}

// SAFETY: see `LaneGeometries`.
unsafe impl<T, D> SplitAt for ArrayElemStridedMut<'_, T, D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let front_len = self.front.len();
        let mid_len = self.geometries.len() * self.lane_len;
        let index = index.min(front_len + mid_len + self.back.len());

        if index <= front_len {
            let (left, right) = self.front.split_at(index);
            let (empty, geometries) = self.geometries.split_at(0);

            (
                Self {
                    access: self.access,
                    geometries: empty,
                    front: left,
                    back: empty_lane_mut(),
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries,
                    front: right,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else if index <= front_len + mid_len {
            let shifted = index - front_len;
            let (left_geoms, mut right_geoms) = self.geometries.split_at(shifted / self.lane_len);
            let mid_split = shifted % self.lane_len;

            let (left_back, right_front) = if mid_split == 0 {
                (empty_lane_mut(), empty_lane_mut())
            } else if let Some(geometry) = right_geoms.next() {
                // SAFETY: the type invariant guarantees that every offset
                // addressed by any geometry of `self.geometries` is guaranteed
                // to be unique and within bounds of the allocation `access`
                // points to.
                let lane = unsafe { LaneElemStridedMut::from_access(self.access, geometry) };

                lane.split_at(mid_split)
            } else {
                (empty_lane_mut(), empty_lane_mut())
            };

            (
                Self {
                    access: self.access,
                    geometries: left_geoms,
                    front: self.front,
                    back: left_back,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: right_geoms,
                    front: right_front,
                    back: self.back,
                    lane_len: self.lane_len,
                },
            )
        } else {
            let shifted = index - front_len - mid_len;
            let (left, right) = self.back.split_at(shifted);
            let geometries_len = self.geometries.len();
            let (geometries, empty) = self.geometries.split_at(geometries_len);

            (
                Self {
                    access: self.access,
                    geometries,
                    front: self.front,
                    back: left,
                    lane_len: self.lane_len,
                },
                Self {
                    access: self.access,
                    geometries: empty,
                    front: right,
                    back: empty_lane_mut(),
                    lane_len: self.lane_len,
                },
            )
        }
    }
}

impl<'s, T, D> ArrayElemStridedMut<'s, T, D>
where
    D: Dimension<Elem = usize>,
{
    /// Creates an iterator over mutable references to elements addressed by
    /// `layout` within `base`.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` if
    /// - [`LaneGeometries::new`] does,
    /// - [`Layout::max_offset`] is not less than the number of elements in
    ///   `base`,
    /// - the layout is not non-overlapping, or
    /// - `layout` and `order` have different ranks.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(base: &'s mut [T], layout: Layout<D>, order: DimOrder<D>) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank()
            || layout.max_offset() >= base.len()
            || !layout.is_non_overlapping()
        {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access: RawAccessMut::from_slice(base),
            geometries,
            front: empty_lane_mut(),
            back: empty_lane_mut(),
            lane_len,
        })
    }

    /// Creates an iterator over mutable references to elements addressed by
    /// `layout` within the allocation `access` points to.
    ///
    /// Elements are yielded according to `order`.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// or if the layout is not non-overlapping, or if `layout` and `order` have
    /// different ranks.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    ///
    /// # Safety
    ///
    /// Every offset layout addresses must be a valid offset into the allocation
    /// `access` points into, and its elements must be borrowed mutably for
    /// `'s`.
    pub(crate) unsafe fn from_access(
        access: RawAccessMut<'s, T>,
        layout: Layout<D>,
        order: DimOrder<D>,
    ) -> Option<Self> {
        let dim = order.last()?;

        if layout.rank() != order.rank() || !layout.is_non_overlapping() {
            return None;
        }

        let lane_len = layout.shape().get(dim)?;
        let geometries = LaneGeometries::new(layout, dim, order)?;

        Some(Self {
            access,
            geometries,
            front: empty_lane_mut(),
            back: empty_lane_mut(),
            lane_len,
        })
    }

    /// Returns the mutable next lane from the front.
    fn next_mut_lane_front(&mut self) -> Option<LaneElemStridedMut<'s, T>> {
        let geometry = self.geometries.next()?;
        // SAFETY: the type invariant guarantees that every offset addressed
        // by any geometry of `self.geometries` is unique and within bounds
        // of the allocation `access` points to.
        let lane = unsafe { LaneElemStridedMut::from_access(self.access, geometry) };

        Some(lane)
    }

    /// Returns the mutable next lane from the back.
    fn next_mut_lane_back(&mut self) -> Option<LaneElemStridedMut<'s, T>> {
        let geometry = self.geometries.next_back()?;
        // SAFETY: see above.
        let lane = unsafe { LaneElemStridedMut::from_access(self.access, geometry) };

        Some(lane)
    }
}

/// Returns an empty lane element iterator.
fn empty_lane<'s, T>() -> LaneElemStrided<'s, T> {
    LaneElemStrided::new(&[], LaneGeometry::new(0, 0, 0).expect("empty is valid"))
        .expect("empty is valid")
}

/// Returns an empty mutable lane element iterator.
fn empty_lane_mut<'s, T>() -> LaneElemStridedMut<'s, T> {
    LaneElemStridedMut::new(&mut [], LaneGeometry::new(0, 0, 0).expect("empty is valid"))
        .expect("empty is valid")
}
