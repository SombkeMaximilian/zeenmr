use crate::intensity_array::iter::{LaneElem, LaneElemStrided, Lanes, ParLaneElemStrided};
use crate::intensity_array::{ArrayIndex, DimIndex, DimOrder, Dimension, DynDim, LaneGeometry, Layout, Shape, StaticDim, Storage, StorageMut, StorageOwned};
use std::borrow::Cow;
use std::ops::{Index, IndexMut};
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{ParLaneElem, ParLanes};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Array borrowing its storage immutably.
pub type ArrayView<'s, T, D> = Array<&'s [T], D>;

/// Array borrowing its storage mutably.
pub type ArrayViewMut<'s, T, D> = Array<&'s mut [T], D>;

/// Array owning its storage.
pub type ArrayOwned<T, D> = Array<Box<[T]>, D>;

/// Array using clone-on-write storage.
pub type ArrayCow<'s, T, D> = Array<Cow<'s, [T]>, D>;

/// Array using referencing counted storage.
pub type ArrayRc<T, D> = Array<Rc<[T]>, D>;

/// Array using atomic reference counted storage.
pub type ArrayArc<T, D> = Array<Arc<[T]>, D>;

/// Array of rank 1.
pub type Array1D<S> = Array<S, StaticDim<1>>;

/// Array of rank 2.
pub type Array2D<S> = Array<S, StaticDim<2>>;

/// Array of rank 3.
pub type Array3D<S> = Array<S, StaticDim<3>>;

/// Array of a rank determined at runtime.
pub type ArrayDyn<S> = Array<S, DynDim>;

/// Array type with arbitrary dimensions.
#[derive(Clone, Debug)]
pub struct Array<S, D> {
    /// Raw storage of the array.
    ///
    /// A `&mut S` must *never* escape to anywhere. Otherwise, all invariants
    /// may be violated, e.g., through a [`Vec::drain`] call.
    storage: S,
    /// Layout of the array.
    ///
    /// Every constructor must establish that [`Layout::max_offset`] is less
    /// than the number of elements in `storage`.
    layout: Layout<D>,
}

impl<S1, D1, S2, D2> PartialEq<Array<S2, D2>> for Array<S1, D1>
where
    S1: Storage,
    S2: Storage,
    S1::Elem: PartialEq<S2::Elem>,
    D1: Dimension,
    D2: Dimension,
{
    // TODO: replace this with fastest lane iteration
    fn eq(&self, other: &Array<S2, D2>) -> bool {
        if self.shape().as_slice() != other.shape().as_slice() {
            return false;
        }

        self.shape()
            .indices_lexicographic()
            .expect("should be validated at construction")
            .all(|index| {
                matches!(
                    (self.get(&index), other.get(&index)),
                    (Some(a), Some(b)) if a == b
                )
            })
    }
}

impl<S, D> Eq for Array<S, D>
where
    S: Storage,
    S::Elem: Eq,
    D: Dimension,
{
}

impl<S, D1, D2> Index<&ArrayIndex<D2>> for Array<S, D1>
where
    S: Storage,
    D1: Dimension,
    D2: Dimension,
{
    type Output = S::Elem;

    #[track_caller]
    fn index(&self, index: &ArrayIndex<D2>) -> &Self::Output {
        if index.rank() != self.rank() {
            index_rank_mismatch(index.rank(), self.rank());
        }
        let Some(linear) = self.layout.linear(index) else {
            index_out_of_bounds(index.as_slice(), self.shape().as_slice());
        };

        // SAFETY: `Layout::linear` returned `Some`, so every component of
        // `index` is less than its extent and the offset is at most
        // `max_offset`.
        unsafe { self.elem_unchecked(linear) }
    }
}

impl<S, D, const N: usize> Index<[usize; N]> for Array<S, D>
where
    S: Storage,
    D: Dimension,
{
    type Output = S::Elem;

    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let index = ArrayIndex::<StaticDim<N>>::from(index);

        &self[&index]
    }
}

impl<S, D1, D2> IndexMut<&ArrayIndex<D2>> for Array<S, D1>
where
    S: StorageMut,
    D1: Dimension,
    D2: Dimension,
{
    #[track_caller]
    fn index_mut(&mut self, index: &ArrayIndex<D2>) -> &mut Self::Output {
        if index.rank() != self.rank() {
            index_rank_mismatch(index.rank(), self.rank());
        }
        let Some(linear) = self.layout.linear(index) else {
            index_out_of_bounds(index.as_slice(), self.shape().as_slice());
        };

        // SAFETY: `Layout::linear` returned `Some`, so every component of
        // `index` is less than its extent and the offset is at most
        // `max_offset`.
        unsafe { self.elem_unchecked_mut(linear) }
    }
}

impl<S, D, const N: usize> IndexMut<[usize; N]> for Array<S, D>
where
    S: StorageMut,
    D: Dimension,
{
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let index = ArrayIndex::<StaticDim<N>>::from(index);

        &mut self[&index]
    }
}

impl<S, D> Array<S, D>
where
    D: Dimension,
{
    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// Returns a reference to the shape of the array.
    pub fn shape(&self) -> &Shape<D> {
        self.layout.shape()
    }

    /// Returns a reference to the layout of the array.
    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }

    /// Returns the number of elements contained in the array.
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// Returns `true` if the array contains no elements.
    ///
    /// This always returns `false`, since no layout constructor produces an
    /// empty layout. It exists as the counterpart to [`Array::len`].
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }
}

impl<S, D> Array<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Creates a row-major, contiguous array with the given shape.
    ///
    /// The elements of `storage` are interpreted in row-major order starting
    /// at its first element. Any trailing elements the shape does not address
    /// are retained but never accessible.
    ///
    /// Returns `None` if [`Layout::row_major`] cannot build a layout from
    /// `shape`, or if `storage` holds fewer elements than that layout
    /// addresses.
    pub fn new(storage: S, shape: Shape<D>) -> Option<Self> {
        Self::from_parts(storage, Layout::row_major(shape, 0)?)
    }

    /// Creates an array from a storage and a layout.
    ///
    /// Returns `None` if `layout` can address an offset past the end of
    /// `storage`, that is if [`Layout::max_offset`] is not less than the
    /// number of elements in `storage`.
    // Every other construct *must* go through here or establish the same
    // invariants about the parts.
    pub fn from_parts(storage: S, layout: Layout<D>) -> Option<Self> {
        if layout.max_offset() >= storage.as_slice().len() {
            return None;
        }

        Some(Self { storage, layout })
    }

    /// Returns an immutable view of the entire array.
    pub fn view(&self) -> ArrayView<'_, S::Elem, D> {
        Array {
            storage: self.storage.as_slice(),
            layout: self.layout.clone(),
        }
    }

    /// Returns the equivalent array with a rank determined at runtime.
    pub fn into_dyn(self) -> Array<S, DynDim> {
        Array {
            storage: self.storage,
            layout: self.layout.into_dyn(),
        }
    }

    /// Returns the equivalent array of rank `N`.
    ///
    /// Returns `Err(self)` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Result<Array<S, StaticDim<N>>, Self> {
        match self.layout.to_dimension() {
            Some(layout) => Ok(Array {
                storage: self.storage,
                layout,
            }),
            None => Err(self),
        }
    }

    /// Returns an iterator over the lanes along `dim` in memory order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn lanes_memory_order(&self, dim: DimIndex) -> Option<Lanes<'_, S::Elem, D>> {
        self.lanes_with_order(dim, self.layout.memory_order())
    }

    /// Returns an iterator over the lanes along `dim` in lexicographic order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn lanes_lexicographic(&self, dim: DimIndex) -> Option<Lanes<'_, S::Elem, D>> {
        self.lanes_with_order(dim, self.layout.lexicographic_order())
    }

    /// Returns an iterator over the lanes along `dim` in the provided order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`.
    pub fn lanes_with_order(
        &self,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<Lanes<'_, S::Elem, D>> {
        Lanes::new(self.storage.as_slice(), self.layout.clone(), dim, order)
    }
}

#[cfg(feature = "rayon")]
impl<S, D> Array<S, D>
where
    S: Storage,
    S::Elem: Sync,
    D: Dimension,
{
    /// Returns a parallel iterator over the lanes along `dim` in memory order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn par_lanes_memory_order(&self, dim: DimIndex) -> Option<ParLanes<'_, S::Elem, D>> {
        self.par_lanes_with_order(dim, self.layout.memory_order())
    }

    /// Returns a parallel iterator over the lanes along `dim` in lexicographic
    /// order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn par_lanes_lexicographic(&self, dim: DimIndex) -> Option<ParLanes<'_, S::Elem, D>> {
        self.par_lanes_with_order(dim, self.layout.lexicographic_order())
    }

    /// Returns a parallel iterator over the lanes along `dim` in the provided
    /// order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`.
    pub fn par_lanes_with_order(
        &self,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<ParLanes<'_, S::Elem, D>> {
        ParLanes::new(self.storage.as_slice(), self.layout.clone(), dim, order)
    }
}

impl<S, D> Array<S, D>
where
    S: StorageMut,
    D: Dimension,
{
    /// Returns a mutable view of the entire array.
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, S::Elem, D> {
        let layout = self.layout.clone();

        Array {
            storage: self.storage.as_mut_slice(),
            layout,
        }
    }
}

impl<S, D> Array<S, D>
where
    S: StorageOwned,
    D: Dimension,
{
    /// Creates a row-major, contiguous array from the linear buffer offsets.
    ///
    /// `f` is called exactly [`Layout::len`] times with the offsets `0..len`
    /// in ascending order, which for the resulting layout visits the elements
    /// in lexicographic index order.
    ///
    /// Returns `None` in the same situations that [`Layout::row_major`] does.
    pub fn from_linear_fn<F>(shape: Shape<D>, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> S::Elem,
    {
        let layout = Layout::row_major(shape, 0)?;
        let storage = (0..layout.len()).map(f).collect();

        Self::from_parts(storage, layout)
    }

    /// Creates a row-major, contiguous array from the multidimensional
    /// indices.
    ///
    /// `f` is called exactly [`Layout::len`] times with every index of the
    /// shape in lexicographic order.
    ///
    /// Returns `None` in the same situations that [`Array::from_linear_fn`]
    /// does.
    pub fn from_index_fn<F>(shape: Shape<D>, mut f: F) -> Option<Self>
    where
        F: FnMut(&ArrayIndex<D>) -> S::Elem,
    {
        let layout = Layout::row_major(shape, 0)?;
        let rank = layout.rank();

        let mut index =
            ArrayIndex::new(D::zero(rank).expect("D can always represent its own rank"));
        let mut data = Vec::with_capacity(layout.len());

        for _ in 0..layout.len() {
            data.push(f(&index));
            index.increment_lexicographic(layout.shape().as_slice());
        }

        Self::from_parts(S::from_vec(data), layout)
    }
}

impl<S, D1> Array<S, D1>
where
    S: Storage,
    D1: Dimension,
{
    /// Returns a reference to the element at `index`.
    ///
    /// Returns `None` if `index` has a different rank than the array, or if
    /// any component is out of bounds.
    pub fn get<D2>(&self, index: &ArrayIndex<D2>) -> Option<&S::Elem>
    where
        D2: Dimension,
    {
        let linear = self.layout.linear(index)?;

        // SAFETY: `Layout::linear` returned `Some`, so every component of
        // `index` is less than its extent and the offset is at most
        // `max_offset`.
        Some(unsafe { self.elem_unchecked(linear) })
    }

    /// Returns a reference to the element at `index`, eliding any checks.
    ///
    /// # Safety
    ///
    /// `index` must have the same rank as the array, and every component must
    /// be less than the corresponding extent.
    pub unsafe fn get_unchecked<D2>(&self, index: &ArrayIndex<D2>) -> &S::Elem
    where
        D2: Dimension,
    {
        // SAFETY: the caller guarantees that `index` is in bounds, so the
        // offset is at most `max_offset`.
        unsafe { self.elem_unchecked(self.layout.linear_unvalidated(index)) }
    }

    /// Returns the lane along `dim` that passes through `index`.
    ///
    /// The component of `index` at `dim` is ignored, so an index anywhere on
    /// the lane selects it.
    ///
    /// This method is a convenience wrapper and returns `None` in the same
    /// situations that [`Layout::lane_at`] does.
    pub fn lane_at<D2>(&self, dim: DimIndex, index: &ArrayIndex<D2>) -> Option<Lane<'_, S::Elem>>
    where
        D2: Dimension,
    {
        let geometry = self.layout.lane_at(dim, index)?;

        Lane::new(self.storage.as_slice(), geometry)
    }

    /// Returns a reference to the element at the given buffer offset.
    ///
    /// # Safety
    ///
    /// `linear` must not exceed [`Layout::max_offset`] of `self.layout`.
    unsafe fn elem_unchecked(&self, linear: usize) -> &S::Elem {
        debug_assert!(linear <= self.layout.max_offset());
        debug_assert!(self.layout.max_offset() < self.storage.as_slice().len());

        // SAFETY: `from_parts` established `max_offset < len`, and `Storage`
        // guarantees `as_slice` keeps returning that same length. The caller
        // guarantees `linear <= max_offset`, hence `linear < len`.
        unsafe { self.storage.as_slice().get_unchecked(linear) }
    }
}

impl<S, D1> Array<S, D1>
where
    S: StorageMut,
    D1: Dimension,
{
    /// Returns a mutable reference to the element at `index`.
    ///
    /// Returns `None` if `index` has a different rank than the array, or if
    /// any component is out of bounds.
    pub fn get_mut<D2>(&mut self, index: &ArrayIndex<D2>) -> Option<&mut S::Elem>
    where
        D2: Dimension,
    {
        let linear = self.layout.linear(index)?;

        // SAFETY: `Layout::linear` returned `Some`, so every component of
        // `index` is less than its extent and the offset is at most
        // `max_offset`.
        Some(unsafe { self.elem_unchecked_mut(linear) })
    }

    /// Returns a mutable reference to the element at `index`, eliding any
    /// checks.
    ///
    /// # Safety
    ///
    /// `index` must have the same rank as the array, and every component must
    /// be less than the corresponding extent.
    pub unsafe fn get_unchecked_mut<D2>(&mut self, index: &ArrayIndex<D2>) -> &mut S::Elem
    where
        D2: Dimension,
    {
        // SAFETY: the caller guarantees that `index` is in bounds, so the
        // offset is at most `max_offset`.
        unsafe { self.elem_unchecked_mut(self.layout.linear_unvalidated(index)) }
    }

    /// Returns a mutable reference to the element at the given buffer offset.
    ///
    /// # Safety
    ///
    /// `linear` must not exceed [`Layout::max_offset`] of `self.layout`.
    #[inline]
    unsafe fn elem_unchecked_mut(&mut self, linear: usize) -> &mut S::Elem {
        debug_assert!(linear <= self.layout.max_offset());
        debug_assert!(self.layout.max_offset() < self.storage.as_slice().len());

        unsafe {
            self.storage
                .as_mut_slice()
                .get_unchecked_mut(linear)
        }
    }
}

/// Lane representation.
#[derive(Copy, Clone, Debug)]
enum LaneInner<S> {
    /// Fastest, memory order dimension gives us contiguous access patterns in
    /// the buffer.
    Contiguous(S),
    /// Elements are `stride` apart in the storage.
    ///
    /// # Safety
    ///
    /// `geometry` must satisfy `geometry.fits_within(base.len())`, so every
    /// offset it addresses is a valid index into `base`. Element access relies
    /// on this to elide bounds checks.
    Strided {
        /// Slice containing the entire storage the lane walks.
        base: S,
        /// Geometry of the lane.
        geometry: LaneGeometry,
    },
}

impl<S> LaneInner<S>
where
    S: Storage,
{
    /// Returns the number of elements in the lane.
    fn len(&self) -> usize {
        match self {
            LaneInner::Contiguous(elements) => elements.as_slice().len(),
            LaneInner::Strided { geometry, .. } => geometry.len(),
        }
    }

    /// Returns `true` if the lane contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    fn is_contiguous(&self) -> bool {
        matches!(self, LaneInner::Contiguous(_))
    }
}

/// Immutable view of a single lane of an array.
#[derive(Debug)]
pub struct Lane<'s, T>(LaneInner<&'s [T]>);

impl<T> Clone for Lane<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Lane<'_, T> {}

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
    /// bounds of the `base`.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        match geometry.contiguous_range() {
            Some(range) => Some(Self(LaneInner::Contiguous(base.get(range)?))),
            None => geometry
                .fits_within(base.len())
                .then_some(Self(LaneInner::Strided { base, geometry })),
        }
    }

    /// Creates a lane view over a contiguous slice.
    pub fn from_slice(elements: &'s [T]) -> Self {
        Self(LaneInner::Contiguous(elements))
    }

    /// Returns the number of elements in the lane.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the lane contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    /// Returns a reference to the `index`-th element of the lane.
    ///
    /// Returns `None` if `index` is not less than [`Lane::len`].
    pub fn get(&self, index: usize) -> Option<&'s T> {
        match self.0 {
            LaneInner::Contiguous(elements) => elements.get(index),
            LaneInner::Strided { base, geometry } => Some(&base[geometry.offset_of(index)?]),
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
                LaneInner::Strided { base, geometry } => {
                    base.get_unchecked(geometry.offset_of_unvalidated(index))
                }
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
    /// should be matched on before a hot path. See [`LaneIterKind`].
    ///
    /// [`LaneIterKind`]: crate::intensity_array::iter::LaneIterKind
    pub fn iter(&self) -> LaneElem<'s, T> {
        match self.0 {
            LaneInner::Contiguous(elements) => LaneElem::Contiguous(elements.iter()),
            LaneInner::Strided { base, geometry } => LaneElem::Strided(
                LaneElemStrided::new(base, geometry).expect("a lane's geometry must be valid"),
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
    /// should be matched on before a hot path. See [`LaneIterKind`].
    ///
    /// [`LaneIterKind`]: crate::intensity_array::iter::LaneIterKind
    pub fn par_iter(&self) -> ParLaneElem<'s, T> {
        match self.0 {
            LaneInner::Contiguous(elements) => ParLaneElem::Contiguous(elements.par_iter()),
            LaneInner::Strided { base, geometry } => ParLaneElem::Strided(
                ParLaneElemStrided::new(base, geometry).expect("a lane's geometry must be valid"),
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
            LaneInner::Strided { base, geometry } => geometry
                .offsets()
                .map(|offset| base[offset].clone())
                .collect(),
        }
    }
}

/// Mutable view of a single lane of an array.
#[derive(Debug)]
pub struct LaneMut<'s, T>(LaneInner<&'s mut [T]>);

impl<'s, T> LaneMut<'s, T> {
    /// Creates a mutable lane view over `base` with the given geometry.
    ///
    /// Returns `None` if any offset computed from `geometry` would be out of
    /// bounds of the `base`.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        match geometry.contiguous_range() {
            Some(range) => Some(Self(LaneInner::Contiguous(base.get_mut(range)?))),
            None => geometry
                .fits_within(base.len())
                .then_some(Self(LaneInner::Strided { base, geometry })),
        }
    }

    /// Creates a mutable lane view over a contiguous slice.
    pub fn from_slice(elements: &'s mut [T]) -> Self {
        Self(LaneInner::Contiguous(elements))
    }

    /// Returns the number of elements in the lane.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the lane contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    /// Returns an immutable view of the same lane.
    pub fn as_lane(&self) -> Lane<'_, T> {
        Lane(match &self.0 {
            LaneInner::Contiguous(base) => LaneInner::Contiguous(base),
            LaneInner::Strided { base, geometry } => LaneInner::Strided {
                base,
                geometry: *geometry,
            },
        })
    }

    /// Converts `self` into an immutable view of the same lane.
    pub fn into_lane(self) -> Lane<'s, T> {
        Lane(match self.0 {
            LaneInner::Contiguous(base) => LaneInner::Contiguous(base),
            LaneInner::Strided { base, geometry } => LaneInner::Strided { base, geometry },
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
            LaneInner::Contiguous(elements) => elements.get_mut(index),
            LaneInner::Strided { base, geometry } => Some(&mut base[geometry.offset_of(index)?]),
        }
    }

    /// Returns a mutable reference to the `index`-th element of the lane.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`LaneMut::len`].
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: the caller guarantees that `index` is less than
        // `LaneMut::len`, which guarantees that it is in bounds.
        unsafe {
            match &mut self.0 {
                LaneInner::Contiguous(elements) => elements.get_unchecked_mut(index),
                LaneInner::Strided { base, geometry } => {
                    base.get_unchecked_mut(geometry.offset_of_unvalidated(index))
                }
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
            LaneInner::Contiguous(elements) => Some(elements),
            LaneInner::Strided { .. } => None,
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
            LaneInner::Contiguous(elements) => Some(elements),
            LaneInner::Strided { .. } => None,
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

/// Panics with a uniform message for an out-of-bounds array index.
#[cold]
#[inline(never)]
#[track_caller]
fn index_out_of_bounds(index: &[usize], shape: &[usize]) -> ! {
    panic!("index {index:?} is out of bounds for array of shape {shape:?}");
}

/// Panics with a uniform message for a rank mismatch.
#[cold]
#[inline(never)]
#[track_caller]
fn index_rank_mismatch(index_rank: usize, array_rank: usize) -> ! {
    panic!("index of rank {index_rank} cannot address array of rank {array_rank}");
}
