use crate::dimension::{DimIndex, Dimension};
use crate::intensity_array::iter::{LaneElemContiguous, LaneGeometries};
use crate::intensity_array::storage::RawAccess;
use crate::intensity_array::{DimOrder, LaneGeometry, Layout};
use crate::iter::SplitAt;
use std::iter::FusedIterator;

/// Iterator over the buffer elements of an array.
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
                // addressed by any geometry of `self.geometries` is guaranteed
                // to be contiguous and within bounds of the allocation `access`
                // points to.
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
    /// Elements are yielded along `dim` and according to `order`.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// or if [`Layout::max_offset`] is not less than the number of elements in
    /// `base`, or if elements along `dim` are not contiguous in the layout.
    ///
    /// Prefer the `elem_*` methods on [`Array`].
    ///
    /// [`Array`]: crate::intensity_array::Array
    pub fn new(
        base: &'s [T],
        layout: Layout<D>,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        if layout.max_offset() >= base.len() || !layout.lanes_are_contiguous(dim) {
            return None;
        }

        let lane_len = layout
            .shape()
            .get(dim)
            .expect("`lanes_are_contiguous` checks that `dim` is in range");
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
    /// Elements are yielded along `dim` and according to `order`.
    ///
    /// Returns `None` in the same situations that [`LaneGeometries::new`] does,
    /// or if elements along `dim` are not contiguous in the layout.
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
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Self> {
        if !layout.lanes_are_contiguous(dim) {
            return None;
        }

        let lane_len = layout
            .shape()
            .get(dim)
            .expect("`lanes_are_contiguous` checks that `dim` is in range");
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
        // SAFETY: `access` and `self.geometries` originate from the same
        // source, meaning they are compatible and every offset addressed by
        // any geometry returned by the latter is guaranteed not to overflow
        // and be within bounds of the allocation `access` points to.
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
    /// and `geometry` are obtained from the same instance of this type.
    ///
    /// # Panics
    ///
    /// Panics if [`LaneGeometry::contiguous_range`] returns `None` for
    /// `geometry`.
    ///
    /// # Safety
    ///
    /// Every offset addressed by `geometry` must not overflow and must be
    /// within bounds of the allocation `access` points to.
    unsafe fn make_slice(access: RawAccess<'s, T>, geometry: LaneGeometry) -> &'s [T] {
        let range = geometry
            .contiguous_range()
            .expect("type guarantees this");

        // SAFETY: the caller guarantees that none of the offsets addressed by
        // `geometry` overflow and that they are within bounds of the allocation
        // `access` points to.
        unsafe { std::slice::from_raw_parts(access.as_ptr().add(range.start), range.len()) }
    }
}
