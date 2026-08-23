use crate::intensity_array::LaneGeometry;
use crate::intensity_array::iter::{LaneElem, LaneElemMut, LaneElemStrided, LaneElemStridedMut};
use crate::intensity_array::storage::{RawAccess, RawAccessMut};

#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{ParLaneElem, ParLaneElemMut};
#[cfg(feature = "rayon")]
use crate::iter::Par;
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};

/// Immutable lane representation.
#[derive(Debug)]
enum LaneInner<'s, T> {
    /// Fastest, memory order dimension gives us contiguous access patterns in
    /// the buffer.
    Contiguous(&'s [T]),
    /// Elements are `stride` apart in the storage.
    ///
    /// # Safety
    ///
    /// All offsets addressable by `geometry` must be valid offsets into the
    /// allocation `access` points to.
    Strided {
        /// Raw access pointer of the storage.
        access: RawAccess<'s, T>,
        /// Geometry of the lane.
        geometry: LaneGeometry,
    },
}

impl<T> Copy for LaneInner<'_, T> {}

impl<T> Clone for LaneInner<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Immutable view of a single lane of an array.
#[derive(Copy, Clone, Debug)]
pub struct Lane<'s, T>(LaneInner<'s, T>);

impl<'s, T> From<LaneMut<'s, T>> for Lane<'s, T> {
    fn from(value: LaneMut<'s, T>) -> Self {
        value.into_lane()
    }
}

impl<'s, T> IntoIterator for Lane<'s, T> {
    type Item = &'s T;
    type IntoIter = LaneElem<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> IntoParallelIterator for Lane<'s, T>
where
    T: Sync,
{
    type Iter = ParLaneElem<'s, T>;
    type Item = &'s T;

    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}

impl<'s, T> Lane<'s, T> {
    /// Creates a lane view over `base` with the given geometry.
    ///
    /// Returns `None` if any offset computed from `geometry` would be out of
    /// bounds of `base`.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        match geometry.contiguous_range() {
            Some(range) => Some(Self(LaneInner::Contiguous(base.get(range)?))),
            None => geometry
                .fits_within(base.len())
                .then_some(Self(LaneInner::Strided {
                    access: RawAccess::from_slice(base),
                    geometry,
                })),
        }
    }

    /// Creates a lane view over a contiguous slice.
    pub fn from_slice(elements: &'s [T]) -> Self {
        Self(LaneInner::Contiguous(elements))
    }

    /// Creates a lane view over the offsets `geometry` addresses relative to
    /// `access`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `access` points into.
    pub(crate) unsafe fn from_access(access: RawAccess<'s, T>, geometry: LaneGeometry) -> Self {
        match geometry.contiguous_range() {
            Some(range) if range.is_empty() => Self(LaneInner::Contiguous(&[])),
            Some(range) => Self(LaneInner::Contiguous(
                // SAFETY: the caller guarantees the entire range lies within a
                // single allocation borrowed for `'s`.
                unsafe {
                    std::slice::from_raw_parts(access.as_ptr().add(range.start), range.len())
                },
            )),
            None => Self(LaneInner::Strided { access, geometry }),
        }
    }

    /// Returns the number of elements in the lane.
    #[inline]
    pub fn len(&self) -> usize {
        match self.0 {
            LaneInner::Contiguous(elements) => elements.len(),
            LaneInner::Strided { geometry, .. } => geometry.len(),
        }
    }

    /// Returns `true` if the lane contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        matches!(self.0, LaneInner::Contiguous(_))
    }

    /// Returns a reference to the `index`-th element of the lane.
    ///
    /// Returns `None` if `index` is not less than [`Lane::len`].
    pub fn get(&self, index: usize) -> Option<&'s T> {
        match self.0 {
            LaneInner::Contiguous(elements) => elements.get(index),
            LaneInner::Strided { access, geometry } => {
                let offset = geometry.offset_of(index)?;

                // SAFETY: `offset_of` returned `Some`, so the offset is one
                // the geometry addresses and therefore in bounds.
                Some(unsafe { &*access.as_ptr().add(offset) })
            }
        }
    }

    /// Returns a reference to the `index`-th element of the lane.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`Lane::len`].
    pub unsafe fn get_unchecked(&self, index: usize) -> &'s T {
        // SAFETY: the caller guarantees that `index` is less than `Lane::len`,
        // which guarantees that it is in bounds.
        unsafe {
            match self.0 {
                LaneInner::Contiguous(elements) => elements.get_unchecked(index),
                LaneInner::Strided { access, geometry } => &*access
                    .as_ptr()
                    .add(geometry.offset_of_unvalidated(index)),
            }
        }
    }

    /// Returns a slice containing the entire lane, or `None` if it isn't
    /// contiguous.
    pub fn as_slice(&self) -> Option<&'s [T]> {
        match self.0 {
            LaneInner::Contiguous(elements) => Some(elements),
            LaneInner::Strided { .. } => None,
        }
    }

    /// Returns an iterator over the elements of the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn iter(&self) -> LaneElem<'s, T> {
        match self.0 {
            LaneInner::Contiguous(elements) => LaneElem::Contiguous(elements.iter()),
            LaneInner::Strided { access, geometry } => LaneElem::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `LaneElemStrided::from_raw` requires.
                unsafe { LaneElemStrided::from_raw(access.as_ptr(), geometry) },
            ),
        }
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> Lane<'s, T>
where
    T: Sync,
{
    /// Returns a parallel iterator over the elements of the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn par_iter(&self) -> ParLaneElem<'s, T> {
        match self.0 {
            LaneInner::Contiguous(elements) => ParLaneElem::Contiguous(elements.par_iter()),
            LaneInner::Strided { access, geometry } => ParLaneElem::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `ParLaneElemStrided::from_raw` requires.
                unsafe { Par::new(LaneElemStrided::from_raw(access.as_ptr(), geometry)) },
            ),
        }
    }
}

impl<'s, T> Lane<'s, T>
where
    T: Clone,
{
    /// Copies the elements of the lane into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        match self.0 {
            LaneInner::Contiguous(elements) => elements.to_vec(),
            LaneInner::Strided { .. } => self.iter().cloned().collect(),
        }
    }
}

/// Mutable lane representation.
#[derive(Debug)]
enum LaneInnerMut<'s, T> {
    /// Fastest, memory order dimension gives us contiguous access patterns in
    /// the buffer.
    Contiguous(&'s mut [T]),
    /// Elements are `stride` apart in the storage.
    ///
    /// # Safety
    ///
    /// All offsets addressable by `geometry` must be valid offsets into the
    /// allocation `access` points to, and `geometry` must be injective.
    Strided {
        /// Raw access pointer of the storage.
        access: RawAccessMut<'s, T>,
        /// Geometry of the lane.
        geometry: LaneGeometry,
    },
}

/// Mutable view of a single lane of an array.
#[derive(Debug)]
pub struct LaneMut<'s, T>(LaneInnerMut<'s, T>);

impl<'s, T> IntoIterator for LaneMut<'s, T> {
    type Item = &'s mut T;
    type IntoIter = LaneElemMut<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            LaneInnerMut::Contiguous(elements) => LaneElemMut::Contiguous(elements.iter_mut()),
            LaneInnerMut::Strided {
                mut access,
                geometry,
            } => LaneElemMut::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `LaneElemStridedMut::from_raw` requires, and consuming `self`
                // transfers the exclusive borrow to the iterator.
                unsafe { LaneElemStridedMut::from_raw(access.as_mut_ptr(), geometry) },
            ),
        }
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> IntoParallelIterator for LaneMut<'s, T>
where
    T: Send + Sync,
{
    type Iter = ParLaneElemMut<'s, T>;
    type Item = &'s mut T;

    fn into_par_iter(self) -> Self::Iter {
        match self.0 {
            LaneInnerMut::Contiguous(elements) => {
                ParLaneElemMut::Contiguous(elements.par_iter_mut())
            }
            LaneInnerMut::Strided {
                mut access,
                geometry,
            } => ParLaneElemMut::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `ParLaneElemStridedMut::from_raw` requires, and consuming
                // `self` transfers the exclusive borrow to the iterator.
                unsafe { Par::new(LaneElemStridedMut::from_raw(access.as_mut_ptr(), geometry)) },
            ),
        }
    }
}

impl<'s, T> LaneMut<'s, T> {
    /// Creates a mutable lane view over `base` with the given geometry.
    ///
    /// Returns `None` if any offset computed from `geometry` would be out of
    /// bounds of the `base`, or if `geometry` is not injective.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        if !geometry.is_injective() {
            return None;
        }

        match geometry.contiguous_range() {
            Some(range) => Some(Self(LaneInnerMut::Contiguous(base.get_mut(range)?))),
            None => geometry
                .fits_within(base.len())
                .then_some(Self(LaneInnerMut::Strided {
                    access: RawAccessMut::from_slice(base),
                    geometry,
                })),
        }
    }

    /// Creates a mutable lane view over a contiguous slice.
    pub fn from_slice(elements: &'s mut [T]) -> Self {
        Self(LaneInnerMut::Contiguous(elements))
    }

    /// Creates a mutable lane view over the offsets `geometry` addresses
    /// relative to `access`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `access` points into, and `geometry` must be injective.
    pub(crate) unsafe fn from_access(
        mut access: RawAccessMut<'s, T>,
        geometry: LaneGeometry,
    ) -> Self {
        match geometry.contiguous_range() {
            Some(range) if range.is_empty() => Self(LaneInnerMut::Contiguous(&mut [])),
            Some(range) => Self(LaneInnerMut::Contiguous(
                // SAFETY: the caller guarantees the entire range lies within a
                // single allocation borrowed for `'s`.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        access.as_mut_ptr().add(range.start),
                        range.len(),
                    )
                },
            )),
            None => Self(LaneInnerMut::Strided { access, geometry }),
        }
    }

    /// Returns the number of elements in the lane.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            LaneInnerMut::Contiguous(elements) => elements.len(),
            LaneInnerMut::Strided { geometry, .. } => geometry.len(),
        }
    }

    /// Returns `true` if the lane contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        matches!(self.0, LaneInnerMut::Contiguous(_))
    }

    /// Returns an immutable view of the same lane.
    pub fn as_lane(&self) -> Lane<'_, T> {
        Lane(match &self.0 {
            LaneInnerMut::Contiguous(elements) => LaneInner::Contiguous(elements),
            // SAFETY: the resulting `Lane` claims nothing outside the offsets
            // `geometry` addresses, which this lane holds exclusively, so it
            // does not overlap a sibling lane.
            LaneInnerMut::Strided { access, geometry } => LaneInner::Strided {
                access: access.as_access(),
                geometry: *geometry,
            },
        })
    }

    /// Converts `self` into an immutable view of the same lane.
    pub fn into_lane(self) -> Lane<'s, T> {
        Lane(match self.0 {
            LaneInnerMut::Contiguous(elements) => LaneInner::Contiguous(elements),
            LaneInnerMut::Strided { access, geometry } => LaneInner::Strided {
                access: access.into_access(),
                geometry,
            },
        })
    }

    /// Returns a mutable view of the same lane borrowing `self`.
    ///
    /// Useful for handing a lane to a function that takes `LaneMut` by value
    /// without giving up the `'s` lifetime borrow.
    pub fn reborrow(&mut self) -> LaneMut<'_, T> {
        LaneMut(match &mut self.0 {
            LaneInnerMut::Contiguous(elements) => LaneInnerMut::Contiguous(elements),
            LaneInnerMut::Strided { access, geometry } => LaneInnerMut::Strided {
                access: access.reborrow(),
                geometry: *geometry,
            },
        })
    }

    /// Returns a reference to the `index`-th element of the lane.
    ///
    /// Returns `None` if `index` is not less than [`LaneMut::len`].
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_lane().get(index)
    }

    /// Returns a reference to the `index`-th element of the lane.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`LaneMut::len`].
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: the caller guarantees that `index` is less than
        // `LaneMut::len`, which guarantees that it is in bounds.
        unsafe { self.as_lane().get_unchecked(index) }
    }

    /// Returns a mutable reference to the `index`-th element of the lane.
    ///
    /// Returns `None` if `index` is not less than [`LaneMut::len`].
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match &mut self.0 {
            LaneInnerMut::Contiguous(elements) => elements.get_mut(index),
            LaneInnerMut::Strided { access, geometry } => {
                let offset = geometry.offset_of(index)?;

                // SAFETY: `offset_of` returned `Some`, so the offset is one
                // the geometry addresses and therefore in bounds, and the
                // `&mut self` borrow makes the reference trivially unique.
                Some(unsafe { &mut *access.as_mut_ptr().add(offset) })
            }
        }
    }

    /// Returns a mutable reference to the `index`-th element of the lane.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`LaneMut::len`].
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: the caller guarantees that `index` is less than
        // `LaneMut::len`, so the resulting offset is in bounds.
        unsafe {
            match &mut self.0 {
                LaneInnerMut::Contiguous(elements) => elements.get_unchecked_mut(index),
                LaneInnerMut::Strided { access, geometry } => &mut *access
                    .as_mut_ptr()
                    .add(geometry.offset_of_unvalidated(index)),
            }
        }
    }

    /// Returns a slice containing the entire lane, or `None` if it isn't
    /// contiguous.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.as_lane().as_slice()
    }

    /// Returns a mutable slice containing the entire lane, or `None` if it
    /// isn't contiguous.
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match &mut self.0 {
            LaneInnerMut::Contiguous(elements) => Some(elements),
            LaneInnerMut::Strided { .. } => None,
        }
    }

    /// Consumes `self` and returns a mutable slice containing the entire lane.
    ///
    /// This method is useful for recovering the original borrow with lifetime
    /// `'s`.
    ///
    /// Returns `None` if `self` isn't contiguous.
    pub fn into_mut_slice(self) -> Option<&'s mut [T]> {
        match self.0 {
            LaneInnerMut::Contiguous(elements) => Some(elements),
            LaneInnerMut::Strided { .. } => None,
        }
    }

    /// Returns an iterator over the elements of the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn iter(&self) -> LaneElem<'_, T> {
        self.as_lane().iter()
    }

    /// Returns an iterator over mutable references of the elements of the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn iter_mut(&mut self) -> LaneElemMut<'_, T> {
        match &mut self.0 {
            LaneInnerMut::Contiguous(elements) => LaneElemMut::Contiguous(elements.iter_mut()),
            LaneInnerMut::Strided { access, geometry } => LaneElemMut::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `LaneElemStridedMut::from_raw` requires, and the `&mut self`
                // borrow  keeps the returned references unique.
                unsafe { LaneElemStridedMut::from_raw(access.as_mut_ptr(), *geometry) },
            ),
        }
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> LaneMut<'s, T>
where
    T: Sync,
{
    /// Returns a parallel iterator over the elements of the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn par_iter(&self) -> ParLaneElem<'_, T> {
        self.as_lane().par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> LaneMut<'s, T>
where
    T: Send + Sync,
{
    /// Returns a parallel iterator over mutable references of the elements of
    /// the lane.
    ///
    /// Note that this is a special iterator type which has two variants that
    /// should be matched on before a hot path. See [`StridedIterKind`].
    ///
    /// [`StridedIterKind`]: crate::intensity_array::iter::StridedIterKind
    pub fn par_iter_mut(&mut self) -> ParLaneElemMut<'_, T> {
        match &mut self.0 {
            LaneInnerMut::Contiguous(elements) => {
                ParLaneElemMut::Contiguous(elements.par_iter_mut())
            }
            LaneInnerMut::Strided { access, geometry } => ParLaneElemMut::Strided(
                // SAFETY: this lane's invariants are exactly the ones
                // `ParLaneElemStridedMut::from_raw` requires, and the
                // `&mut self` borrow  keeps the returned references unique.
                unsafe { Par::new(LaneElemStridedMut::from_raw(access.as_mut_ptr(), *geometry)) },
            ),
        }
    }
}

impl<'s, T> LaneMut<'s, T>
where
    T: Clone,
{
    /// Copies the elements of the lane into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_lane().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(Lane<'_, u8>: Send, Sync);
        assert_impl_all!(LaneInner<'_, u8>: Send, Sync);
        assert_impl_all!(LaneMut<'_, u8>: Send, Sync);
        assert_impl_all!(LaneInnerMut<'_, u8>: Send, Sync);
    }

    #[test]
    fn non_injective_geometry() {
        let mut storage = vec![0_i32; 4];
        let geometry = LaneGeometry::new(0, 0, 3).expect("hand verified");
        let lane = Lane::new(&storage, geometry).expect("hand verified");

        assert_eq!(lane.iter().count(), 3);
        assert!(!lane.is_contiguous());
        assert!(LaneMut::new(&mut storage, geometry).is_none());
        assert!(LaneElemStridedMut::new(&mut storage, geometry).is_none());
    }

    #[test]
    fn oversized_geometry() {
        let geometry = LaneGeometry::new(0, 3, 3).expect("hand verified");
        let mut short = vec![0_i32; 6];
        let mut exact = vec![0_i32; 7];
        let mut long = vec![0_i32; 100];

        assert_eq!(geometry.max_offset(), Some(6));
        assert!(Lane::new(&short, geometry).is_none());
        assert!(Lane::new(&exact, geometry).is_some());
        assert!(Lane::new(&long, geometry).is_some());
        assert!(LaneMut::new(&mut short, geometry).is_none());
        assert!(LaneMut::new(&mut exact, geometry).is_some());
        assert!(LaneMut::new(&mut long, geometry).is_some());
        assert!(LaneElemStrided::new(&short, geometry).is_none());
        assert!(LaneElemStrided::new(&exact, geometry).is_some());
        assert!(LaneElemStrided::new(&long, geometry).is_some());
        assert!(LaneElemStridedMut::new(&mut short, geometry).is_none());
        assert!(LaneElemStridedMut::new(&mut exact, geometry).is_some());
        assert!(LaneElemStridedMut::new(&mut long, geometry).is_some());
    }
}
