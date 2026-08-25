use crate::dimension::{DimIndex, Dimension, DynDim, StaticDim, assert_rank_compatible};
use crate::intensity_array::iter::{Lanes, LanesMut};
use crate::intensity_array::storage::{RawAccess, RawAccessMut};
use crate::intensity_array::{
    ArrayIndex, DimOrder, Lane, LaneMut, Layout, RawStorage, RawStorageMut, Shape, Storage,
    StorageMut, StorageOwned,
};
use std::borrow::Cow;
use std::ops::{Index, IndexMut, RangeBounds};
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{ParLanes, ParLanesMut};
#[cfg(feature = "rayon")]
use crate::iter::Par;

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
pub type Array1D<S> = Array<S, StaticDim<usize, 1>>;

/// Array of rank 2.
pub type Array2D<S> = Array<S, StaticDim<usize, 2>>;

/// Array of rank 3.
pub type Array3D<S> = Array<S, StaticDim<usize, 3>>;

/// Array of a rank determined at runtime.
pub type ArrayDyn<S> = Array<S, DynDim<usize>>;

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
    /// Every constructor must establish that [`Layout::max_offset`] is a valid
    /// offset into the allocation `storage` points into, and thereby that every
    /// smaller offset is too. Further, the elements the layout addresses must
    /// be borrowed for as long as `storage` borrows them, exclusively when
    /// `S: RawStorageMut` or `S: StorageMut`.
    layout: Layout<D>,
}

impl<S1, D1, S2, D2> PartialEq<Array<S2, D2>> for Array<S1, D1>
where
    S1: RawStorage,
    S2: RawStorage,
    S1::Elem: PartialEq<S2::Elem>,
    D1: Dimension<Elem = usize>,
    D2: Dimension<Elem = usize>,
{
    // TODO: replace this with fastest lane iteration
    fn eq(&self, other: &Array<S2, D2>) -> bool {
        const { assert_rank_compatible::<D1, D2>() };

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
    S: RawStorage,
    S::Elem: Eq,
    D: Dimension<Elem = usize>,
{
}

impl<S, D1, D2> Index<ArrayIndex<D2>> for Array<S, D1>
where
    S: RawStorage,
    D1: Dimension<Elem = usize>,
    D2: Dimension<Elem = usize>,
{
    type Output = S::Elem;

    #[track_caller]
    fn index(&self, index: ArrayIndex<D2>) -> &Self::Output {
        const { assert_rank_compatible::<D1, D2>() };

        self.index(&index)
    }
}

impl<S, D1, D2> Index<&ArrayIndex<D2>> for Array<S, D1>
where
    S: RawStorage,
    D1: Dimension<Elem = usize>,
    D2: Dimension<Elem = usize>,
{
    type Output = S::Elem;

    #[track_caller]
    fn index(&self, index: &ArrayIndex<D2>) -> &Self::Output {
        const { assert_rank_compatible::<D1, D2>() };
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
    S: RawStorage,
    D: Dimension<Elem = usize>,
{
    type Output = S::Elem;

    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let index = ArrayIndex::<StaticDim<usize, N>>::from(index);

        &self[&index]
    }
}

impl<S, D1, D2> IndexMut<ArrayIndex<D2>> for Array<S, D1>
where
    S: RawStorageMut,
    D1: Dimension<Elem = usize>,
    D2: Dimension<Elem = usize>,
{
    #[track_caller]
    fn index_mut(&mut self, index: ArrayIndex<D2>) -> &mut Self::Output {
        const { assert_rank_compatible::<D1, D2>() };

        self.index_mut(&index)
    }
}

impl<S, D1, D2> IndexMut<&ArrayIndex<D2>> for Array<S, D1>
where
    S: RawStorageMut,
    D1: Dimension<Elem = usize>,
    D2: Dimension<Elem = usize>,
{
    #[track_caller]
    fn index_mut(&mut self, index: &ArrayIndex<D2>) -> &mut Self::Output {
        const { assert_rank_compatible::<D1, D2>() };
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
    S: RawStorageMut,
    D: Dimension<Elem = usize>,
{
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let index = ArrayIndex::<StaticDim<usize, N>>::from(index);

        &mut self[&index]
    }
}

impl<S, D> Array<S, D>
where
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
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
}

impl<S, D> Array<S, D>
where
    S: StorageMut,
    D: Dimension<Elem = usize>,
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
    D: Dimension<Elem = usize>,
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
            ArrayIndex::new(D::from_fn(rank, |_| 0).expect("D can always represent its own rank"));
        let mut data = Vec::with_capacity(layout.len());

        for _ in 0..layout.len() {
            data.push(f(&index));
            index.increment_lexicographic(layout.shape().as_slice());
        }

        Self::from_parts(S::from_vec(data), layout)
    }
}

impl<S, D> Array<S, D>
where
    S: RawStorage,
    D: Dimension<Elem = usize>,
{
    /// Returns the equivalent array with a rank determined at runtime.
    pub fn into_dyn(self) -> Array<S, DynDim<usize>> {
        Array {
            storage: self.storage,
            layout: self.layout.to_dyn(),
        }
    }

    /// Returns the equivalent array of rank `N`.
    ///
    /// Returns `Err(self)` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Result<Array<S, StaticDim<usize, N>>, Self> {
        match self.layout.to_dimension() {
            Some(layout) => Ok(Array {
                storage: self.storage,
                layout,
            }),
            None => Err(self),
        }
    }

    /// Restricts the extent along `dim` to 1 at `index`.
    ///
    /// Returns `None` if `dim` is out of range, or if `index` is not less
    /// than the extent along `dim`. In any such case, `self` remains
    /// unmodified.
    pub fn restrict(&mut self, dim: DimIndex, index: usize) -> Option<&mut Self> {
        self.layout.restrict(dim, index)?;

        Some(self)
    }

    /// Restricts the extent along `dim` to `range`.
    ///
    /// Returns `None` if `dim` is out of range, or if `range` is empty or not
    /// contained in `0..extent`. In any such case, `self` remains unmodified.
    pub fn crop<R>(&mut self, dim: DimIndex, range: R) -> Option<&mut Self>
    where
        R: RangeBounds<usize>,
    {
        self.layout.crop(dim, range)?;

        Some(self)
    }

    /// Returns the dimension along which lanes are contiguous, together with
    /// an iterator over those lanes in memory order.
    ///
    /// Every lane the iterator yields is contiguous, so [`Lane::as_slice`]
    /// returns `Some` for all of them.
    ///
    /// Returns `None` if no dimension has stride 1.
    pub fn contiguous_lanes(&self) -> Option<(DimIndex, Lanes<'_, S::Elem, D>)> {
        let dim = self.layout.contiguous_dimension()?;

        Some((dim, self.lanes_memory_order(dim)?))
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
        // SAFETY: `RawStorage` guarantees the base pointer is non-null and
        // aligned, and the type invariant guarantees every offset the layout
        // addresses is valid. The returned lanes borrow `self`, so the elements
        // stay borrowed for as long as the access pointer exists.
        unsafe {
            Lanes::from_access(
                RawAccess::from_raw(self.storage.as_ptr()),
                self.layout.clone(),
                dim,
                order,
            )
        }
    }
}

impl<S> Array<S, DynDim<usize>>
where
    S: RawStorage,
{
    /// Drops dimensions of extent 1 from the array.
    ///
    /// The rank of the array reduces by the number of dropped dimensionas.
    /// Every other property of the array remains unchanged.
    ///
    /// Note that any `DimIndex` created prior to this method call potentially
    /// becomes meaningless.
    pub fn drop_unit_extents(&mut self) -> &mut Self {
        self.layout.drop_unit_extents();

        self
    }
}

impl<S, D> Array<S, D>
where
    S: RawStorage,
    D: Dimension<Elem = usize>,
    Array<S, D>: Clone,
{
    /// Returns the array with dimensions of extent 1 dropped.
    ///
    /// For owned storage, this clones the entire buffer. Converting the array
    /// to a dynamic one with [`Array::into_dyn`], and then calling
    /// [`Array::drop_unit_extents`] on the result will avoid the clone.
    /// Alternatively, acquiring a view via [`Array::view`] and calling this
    /// method on the result preserves the static array.
    ///
    /// [`Array::drop_unit_extents`]: ArrayDyn::drop_unit_extents
    ///
    /// The rank of the array reduces by the number of dropped dimensions.
    /// Every other property of the array remains unchanged.
    ///
    /// Note that any `DimIndex` created prior to this method call potentially
    /// becomes meaningless.
    pub fn unit_extents_dropped(&self) -> Array<S, DynDim<usize>> {
        let mut array = self.clone().into_dyn();
        array.drop_unit_extents();

        array
    }

    /// Returns the array with its extent along `dim` restricted to 1 at
    /// `index`.
    ///
    /// For owned storage, this clones the entire buffer. Acquiring a view via
    /// [`Array::view`] and calling this method on the result avoids the clone.
    /// This operation can also be performed in place through
    /// [`Array::restrict`].
    ///
    /// Returns `None` if `dim` is out of range, or if `index` is not less
    /// than the extent along `dim`.
    pub fn restricted(&self, dim: DimIndex, index: usize) -> Option<Self> {
        let mut array = self.clone();
        array.restrict(dim, index)?;

        Some(array)
    }

    /// Returns the array with its extent along `dim` restricted to `range`.
    ///
    /// For owned storage, this clones the entire buffer. Acquiring a view via
    /// [`Array::view`] and calling this method on the result avoids the clone.
    /// This operation can also be performed in place with [`Array::crop`].
    ///
    /// Returns `None` if `dim` is out of range, or if `range` is empty or not
    /// contained in `0..extent`.
    pub fn cropped<R>(&self, dim: DimIndex, range: R) -> Option<Self>
    where
        R: RangeBounds<usize>,
    {
        let mut array = self.clone();
        array.crop(dim, range)?;

        Some(array)
    }
}

impl<S, D1> Array<S, D1>
where
    S: RawStorage,
    D1: Dimension<Elem = usize>,
    Array<S, D1>: Clone,
{
    /// Returns the array with its dimensions reordered according to `order`.
    ///
    /// For owned storage, this clones the entire buffer. Acquiring a view via
    /// [`Array::view`] and calling this method on the result avoids the clone.
    /// This operation can also be performed in place with [`Array::permute`].
    ///
    /// Dimension `i` of the result is dimension `order[i]` of `self`, for both
    /// extents and strides. Permuting by [`Layout::memory_order`] yields the
    /// layout whose lexicographic traversal is the most sequential one.
    ///
    /// Returns `None` if `order` has a different rank than `self`.
    pub fn permuted<D2>(&self, order: &DimOrder<D2>) -> Option<Self>
    where
        D2: Dimension<Elem = usize>,
    {
        let mut array = self.clone();
        array.permute(order)?;

        Some(array)
    }
}

#[cfg(feature = "rayon")]
impl<S, D> Array<S, D>
where
    S: RawStorage,
    S::Elem: Sync,
    D: Dimension<Elem = usize>,
{
    /// Returns the dimension along which lanes are contiguous, together with
    /// a parallel iterator over those lanes in memory order.
    ///
    /// Every lane the iterator yields is contiguous, so
    /// [`Lane::as_slice`] returns `Some` for all of them.
    ///
    /// Returns `None` if no dimension has stride 1.
    pub fn par_contiguous_lanes(&self) -> Option<(DimIndex, ParLanes<'_, S::Elem, D>)> {
        let dim = self.layout.contiguous_dimension()?;

        Some((dim, self.par_lanes_memory_order(dim)?))
    }

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
        self.lanes_with_order(dim, order).map(Par::new)
    }
}

impl<S, D> Array<S, D>
where
    S: RawStorageMut,
    D: Dimension<Elem = usize>,
{
    /// Returns the dimension along which lanes are contiguous, together with
    /// an iterator over those mutable lanes in memory order.
    ///
    /// Every lane the iterator yields is contiguous, so
    /// [`LaneMut::as_mut_slice`] returns `Some` for all of them.
    ///
    /// Returns `None` if no dimension has stride 1, or if the layout of `self`
    /// is self-overlapping.
    pub fn contiguous_lanes_mut(&mut self) -> Option<(DimIndex, LanesMut<'_, S::Elem, D>)> {
        let dim = self.layout.contiguous_dimension()?;

        Some((dim, self.lanes_memory_order_mut(dim)?))
    }

    /// Returns an iterator over the mutable lanes along `dim` in memory order.
    ///
    /// Returns `None` if `dim` is out of range, or if the layout of `self` is
    /// self-overlapping.
    pub fn lanes_memory_order_mut(&mut self, dim: DimIndex) -> Option<LanesMut<'_, S::Elem, D>> {
        self.lanes_with_order_mut(dim, self.layout.memory_order())
    }

    /// Returns an iterator over the mutable lanes along `dim` in lexicographic
    /// order.
    ///
    /// Returns `None` if `dim` is out of range, or if the layout of `self` is
    /// self-overlapping.
    pub fn lanes_lexicographic_mut(&mut self, dim: DimIndex) -> Option<LanesMut<'_, S::Elem, D>> {
        self.lanes_with_order_mut(dim, self.layout.lexicographic_order())
    }

    /// Returns an iterator over the mutable lanes along `dim` in the provided
    /// order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`, or if the layout of `self` is self-overlapping.
    pub fn lanes_with_order_mut(
        &mut self,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<LanesMut<'_, S::Elem, D>> {
        // SAFETY: `RawStorageMut` guarantees the base pointer is non-null and
        // aligned, and the type invariant guarantees every offset the layout
        // addresses is valid. The returned lanes borrow `self` mutably, so no
        // other reference to the storage can exist while they do.
        unsafe {
            LanesMut::from_access(
                RawAccessMut::from_raw(self.storage.as_mut_ptr()),
                self.layout.clone(),
                dim,
                order,
            )
        }
    }
}

#[cfg(feature = "rayon")]
impl<S, D> Array<S, D>
where
    S: RawStorageMut,
    S::Elem: Send,
    D: Dimension<Elem = usize>,
{
    /// Returns the dimension along which lanes are contiguous, together with
    /// a parallel iterator over those mutable lanes in memory order.
    ///
    /// Every lane the iterator yields is contiguous, so
    /// [`LaneMut::as_mut_slice`] returns `Some` for all of them.
    ///
    /// Returns `None` if no dimension has stride 1, or if the layout of `self`
    /// is self-overlapping.
    pub fn par_contiguous_lanes_mut(&mut self) -> Option<(DimIndex, ParLanesMut<'_, S::Elem, D>)> {
        let dim = self.layout.contiguous_dimension()?;

        Some((dim, self.par_lanes_memory_order_mut(dim)?))
    }

    /// Returns a parallel iterator over the mutable lanes along `dim` in memory
    /// order.
    ///
    /// Returns `None` if `dim` is out of range, or if the layout of `self` is
    /// self-overlapping.
    pub fn par_lanes_memory_order_mut(
        &mut self,
        dim: DimIndex,
    ) -> Option<ParLanesMut<'_, S::Elem, D>> {
        self.par_lanes_with_order_mut(dim, self.layout.memory_order())
    }

    /// Returns a parallel iterator over the mutable lanes along `dim` in
    /// lexicographic order.
    ///
    /// Returns `None` if `dim` is out of range, or if the layout of `self` is
    /// self-overlapping.
    pub fn par_lanes_lexicographic_mut(
        &mut self,
        dim: DimIndex,
    ) -> Option<ParLanesMut<'_, S::Elem, D>> {
        self.par_lanes_with_order_mut(dim, self.layout.lexicographic_order())
    }

    /// Returns a parallel iterator over the mutable lanes along `dim` in the
    /// provided order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`, or if the layout of `self` is self-overlapping.
    pub fn par_lanes_with_order_mut(
        &mut self,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<ParLanesMut<'_, S::Elem, D>> {
        self.lanes_with_order_mut(dim, order)
            .map(Par::new)
    }
}

impl<S, D1> Array<S, D1>
where
    S: RawStorage,
    D1: Dimension<Elem = usize>,
{
    /// Reorders the dimensions accroding to `order`.
    ///
    /// Dimension `i` of the result is dimension `order[i]` of `self`, for both
    /// extents and strides. Permuting by [`Layout::memory_order`] yields the
    /// layout whose lexicographic traversal is the most sequential one.
    ///
    /// Returns `None` if `order` has a different rank than `self`, in which
    /// case `self` remains unmodified.
    pub fn permute<D2>(&mut self, order: &DimOrder<D2>) -> Option<&mut Self>
    where
        D2: Dimension<Elem = usize>,
    {
        self.layout.permute(order)?;

        Some(self)
    }

    /// Returns a reference to the element at `index`.
    ///
    /// Returns `None` if `index` has a different rank than the array, or if
    /// any component is out of bounds.
    pub fn get<D2>(&self, index: &ArrayIndex<D2>) -> Option<&S::Elem>
    where
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

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
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

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
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

        let geometry = self.layout.lane_at(dim, index)?;

        // SAFETY: `RawStorage` guarantees the base pointer is non-null and
        // aligned, and the type invariant guarantees every offset the layout
        // addresses is valid. The returned lanes borrow `self`, so the elements
        // stay borrowed for as long as the access pointer exists.
        Some(unsafe { Lane::from_access(RawAccess::from_raw(self.storage.as_ptr()), geometry) })
    }

    /// Returns a reference to the element at the given buffer offset.
    ///
    /// # Safety
    ///
    /// `linear` must not exceed [`Layout::max_offset`] of `self.layout`.
    unsafe fn elem_unchecked(&self, linear: usize) -> &S::Elem {
        debug_assert!(linear <= self.layout.max_offset());

        // SAFETY: the type invariant establishes that `max_offset` is a valid
        // index into the storage allocation, and the caller guarantees
        // `linear <= max_offset`. `RawStorage` guarantees the base pointer is
        // non-null and aligned, so the offset pointer is dereferenceable, and
        // the `&self` borrow keeps the reference valid.
        unsafe { &*self.storage.as_ptr().add(linear) }
    }
}

impl<S, D1> Array<S, D1>
where
    S: RawStorageMut,
    D1: Dimension<Elem = usize>,
{
    /// Returns a mutable reference to the element at `index`.
    ///
    /// Returns `None` if `index` has a different rank than the array, or if
    /// any component is out of bounds.
    pub fn get_mut<D2>(&mut self, index: &ArrayIndex<D2>) -> Option<&mut S::Elem>
    where
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

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
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

        // SAFETY: the caller guarantees that `index` is in bounds, so the
        // offset is at most `max_offset`.
        unsafe { self.elem_unchecked_mut(self.layout.linear_unvalidated(index)) }
    }

    /// Returns the mutable lane along `dim` that passes through `index`.
    ///
    /// The component of `index` at `dim` is ignored, so an index anywhere on
    /// the lane selects it.
    ///
    /// This method is a convenience wrapper and returns `None` in the same
    /// situations that [`Layout::lane_at`] does, or if the geometry returned
    /// by layout of `self` is non-injective for that lane.
    pub fn lane_at_mut<D2>(
        &mut self,
        dim: DimIndex,
        index: &ArrayIndex<D2>,
    ) -> Option<LaneMut<'_, S::Elem>>
    where
        D2: Dimension<Elem = usize>,
    {
        const { assert_rank_compatible::<D1, D2>() };

        let geometry = self.layout.lane_at(dim, index)?;

        // SAFETY: `RawStorageMut` guarantees the base pointer is non-null and
        // aligned, and the type invariant guarantees every offset the layout
        // addresses is valid. The returned lane borrows `self` mutably, so no
        // other reference to the storage can exist while they do.
        Some(unsafe {
            LaneMut::from_access(RawAccessMut::from_raw(self.storage.as_mut_ptr()), geometry)
        })
    }

    /// Returns a mutable reference to the element at the given buffer offset.
    ///
    /// # Safety
    ///
    /// `linear` must not exceed [`Layout::max_offset`] of `self.layout`.
    unsafe fn elem_unchecked_mut(&mut self, linear: usize) -> &mut S::Elem {
        debug_assert!(linear <= self.layout.max_offset());

        // SAFETY: the type invariant establishes that `max_offset` is a valid
        // index into the storage allocation, and the caller guarantees
        // `linear <= max_offset`. `RawStorage` guarantees the base pointer is
        // non-null and aligned, so the offset pointer is dereferenceable, and
        // the `&mut self` borrow keeps the reference valid.
        unsafe { &mut *self.storage.as_mut_ptr().add(linear) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intensity_array::iter::{LaneElemStrided, LaneElemStridedMut};
    use crate::intensity_array::{AccessMut, Strides};
    use static_assertions::assert_not_impl_any;

    #[test]
    fn mutable_borrows_not_clone() {
        assert_not_impl_any!(ArrayViewMut<'_, u8, StaticDim<usize, 1>>: Copy, Clone);
        assert_not_impl_any!(ArrayViewMut<'_, u8, DynDim<usize>>: Copy, Clone);
        assert_not_impl_any!(Array<AccessMut<'_, u8>, StaticDim<usize, 1>>: Copy, Clone);
        assert_not_impl_any!(Array<AccessMut<'_, u8>, DynDim<usize>>: Copy, Clone);
    }

    #[test]
    fn from_parts_boundary() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let exact = vec![0_i32; 6].into_boxed_slice();
        let short = vec![0_i32; 5].into_boxed_slice();

        assert_eq!(layout.max_offset(), 5);
        assert!(Array::from_parts(exact, layout.clone()).is_some());
        assert!(Array::from_parts(short, layout.clone()).is_none());

        let long = (0..100).collect::<Box<[i32]>>();
        let array = Array::from_parts(long, layout.clone()).expect("hand verified");

        assert_eq!(array.len(), 6);
        assert_eq!(array.layout.max_offset(), 5);
        assert_eq!(
            array.get(&ArrayIndex::new(DynDim::from_array([1, 2]))),
            Some(&5)
        );
        assert_eq!(
            array.get(&ArrayIndex::new(DynDim::from_array([1, 3]))),
            None
        );
        assert_eq!(
            array.get(&ArrayIndex::new(DynDim::from_array([2, 2]))),
            None
        );
        assert_eq!(
            array.get(&ArrayIndex::new(DynDim::from_array([2, 3]))),
            None
        );
    }

    #[test]
    fn array_validation() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let layout = Layout::row_major(shape, 2).expect("hand verified");
        let storage = (0..8).collect::<Box<[i32]>>();
        let array = Array::from_parts(storage, layout).expect("hand verified");
        let indices = (
            ArrayIndex::new(DynDim::from_array([0, 0])),
            ArrayIndex::new(DynDim::from_array([1, 2])),
        );

        assert_eq!(array.get(&indices.0), Some(&2));
        assert_eq!(array.get(&indices.1), Some(&7));
        assert_eq!(array.len(), 6);
        assert_eq!(array.layout.max_offset(), 7);

        let short = (0..7).collect::<Box<[i32]>>();
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let layout = Layout::row_major(shape, 2).expect("hand verified");

        assert!(Array::from_parts(short, layout).is_none());
    }

    #[test]
    fn owned_construction() {
        fn owned_construction_<S>()
        where
            S: StorageOwned<Elem = i32> + std::fmt::Debug,
        {
            let shape = Shape::new(DynDim::from_array([2, 3, 2]));
            let linear =
                Array::<S, _>::from_linear_fn(shape.clone(), |i| i as i32).expect("hand verified");
            let indexed = Array::<S, _>::from_index_fn(shape.clone(), |i| {
                let i = i.as_slice();

                (i[0] * 6 + i[1] * 2 + i[2]) as i32
            })
            .expect("hand verified");

            assert_eq!(linear, indexed);
            assert_eq!(linear[[1, 2, 1]], 11);
        }

        owned_construction_::<Vec<i32>>();
        owned_construction_::<Box<[i32]>>();
        owned_construction_::<Cow<'static, [i32]>>();
        owned_construction_::<Rc<[i32]>>();
        owned_construction_::<Arc<[i32]>>();
    }

    #[test]
    fn owned_construction_order() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let mut order = Vec::new();
        let _ = ArrayOwned::from_linear_fn(shape.clone(), |i| {
            order.push(i);

            0_i32
        })
        .expect("hand verified");

        assert_eq!(order, vec![0, 1, 2, 3, 4, 5]);

        let mut order = Vec::new();
        let _ = ArrayOwned::from_index_fn(shape.clone(), |i| {
            order.push(i.as_slice().to_vec());

            0_i32
        })
        .expect("hand verified");

        assert_eq!(
            order,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ],
        );
    }

    #[test]
    fn clone_on_write() {
        let borrowed = [0, 1, 2, 3, 4, 5];
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let mut array = Array::new(Cow::Borrowed(&borrowed[..]), shape).expect("hand verified");
        let mutated = ArrayIndex::new(DynDim::from_array([1, 2]));
        *array.get_mut(&mutated).expect("in bounds") = 99;

        assert_eq!(array[[0, 0]], 0);
        assert_eq!(array[[0, 1]], 1);
        assert_eq!(array[[0, 2]], 2);
        assert_eq!(array[[1, 0]], 3);
        assert_eq!(array[[1, 1]], 4);
        assert_eq!(array[&mutated], 99);
        for (i, value) in borrowed.into_iter().enumerate() {
            assert_eq!(value, i as i32);
        }
    }

    #[test]
    fn shared_storage() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let array = ArrayArc::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let clone = array.clone();

        assert_eq!(clone, array);
    }

    #[test]
    fn zero_extents() {
        let storage = (0..6).collect::<Box<[i32]>>();

        for extents in [[6, 0], [0, 6], [0, 0]] {
            let shape = Shape::new(DynDim::from_array(extents));

            assert!(Array::new(storage.as_ref(), shape).is_none());
        }
    }

    #[test]
    fn rank_zero_array() {
        let zero = DynDim::from_array([]);
        let shape = Shape::new(zero);
        let array = Array::new(vec![0].into_boxed_slice(), shape.clone()).expect("hand verified");

        assert_eq!(array.rank(), 0);
        assert_eq!(array.len(), 1);
        assert!(!array.is_empty());
        assert_eq!(array[[]], 0);
        assert_eq!(
            array.get(&ArrayIndex::new(DynDim::from_array([]))),
            Some(&0)
        );
        assert_eq!(array.get(&ArrayIndex::new(DynDim::from_array([0]))), None);
        assert!(array.contiguous_lanes().is_none());
        assert!(array.lanes_memory_order(DimIndex(0)).is_none());
    }

    #[test]
    fn cropped_array() {
        let shape = Shape::new(DynDim::from_array([5, 5, 10]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        assert!(array.cropped(DimIndex(0), 2..4).is_some());
        assert!(array.cropped(DimIndex(0), 2..).is_some());
        assert!(array.cropped(DimIndex(0), ..4).is_some());
        assert!(array.cropped(DimIndex(0), ..).is_some());
        assert!(array.cropped(DimIndex(0), 2..6).is_none());
        assert!(array.cropped(DimIndex(0), 2..2).is_none());

        let cropped = array
            .view()
            .cropped(DimIndex(0), 2..4)
            .and_then(|array| array.cropped(DimIndex(1), 3..5))
            .and_then(|array| array.cropped(DimIndex(2), ..3))
            .expect("all in bounds");

        assert_eq!(
            cropped,
            Array::from_parts(
                vec![130, 131, 132, 140, 141, 142, 180, 181, 182, 190, 191, 192],
                Layout::row_major(cropped.shape().clone(), 0).expect("hand verified"),
            )
            .expect("hand verified"),
        );

        let mut cropped_in_place = array.view();
        cropped_in_place
            .crop(DimIndex(0), 2..4)
            .and_then(|array| array.crop(DimIndex(1), 3..5))
            .and_then(|array| array.crop(DimIndex(2), ..3))
            .expect("all in bounds");

        assert_eq!(cropped, cropped_in_place);
        assert_eq!(cropped.shape(), cropped_in_place.shape());
        assert_eq!(cropped.layout(), cropped_in_place.layout());
    }

    #[test]
    fn restricted_array() {
        let shape = Shape::new(DynDim::from_array([5, 5, 10]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        let restricted = array
            .view()
            .restricted(DimIndex(0), 4)
            .and_then(|array| array.restricted(DimIndex(1), 1))
            .expect("all in bounds");

        assert_eq!(
            restricted,
            Array::from_parts(
                (210..220).collect::<Vec<i32>>(),
                Layout::row_major(restricted.shape().clone(), 0).expect("hand verified"),
            )
            .expect("hand verified"),
        );

        let mut restricted_in_place = array.view();
        restricted_in_place
            .restrict(DimIndex(0), 4)
            .and_then(|array| array.restrict(DimIndex(1), 1))
            .expect("all in bounds");

        assert_eq!(restricted, restricted_in_place);
        assert_eq!(restricted.shape(), restricted_in_place.shape());
        assert_eq!(restricted.layout(), restricted_in_place.layout());
    }

    #[test]
    fn drop_unit_extents() {
        let shape = Shape::new(DynDim::from_array([5, 5, 10]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        let rank_one = array
            .view()
            .cropped(DimIndex(0), 3..5)
            .and_then(|array| array.restricted(DimIndex(1), 2))
            .and_then(|array| array.restricted(DimIndex(2), 2))
            .map(|array| array.unit_extents_dropped())
            .expect("all in bounds");

        assert_eq!(
            rank_one,
            Array::from_parts(
                vec![172, 222],
                Layout::row_major(rank_one.shape().clone(), 0).expect("hand verified"),
            )
            .expect("hand verified"),
        );

        let mut rank_one_in_place = array.view();
        rank_one_in_place
            .crop(DimIndex(0), 3..5)
            .and_then(|array| array.restrict(DimIndex(1), 2))
            .and_then(|array| array.restrict(DimIndex(2), 2))
            .map(|array| array.drop_unit_extents())
            .expect("all in bounds");

        assert_eq!(rank_one, rank_one_in_place);
        assert_eq!(rank_one.shape(), rank_one_in_place.shape());
        assert_eq!(rank_one.layout(), rank_one_in_place.layout());

        let rank_zero = array
            .view()
            .restrict(DimIndex(0), 4)
            .and_then(|array| array.restrict(DimIndex(1), 2))
            .and_then(|array| array.restrict(DimIndex(2), 1))
            .map(|array| array.unit_extents_dropped())
            .expect("all in bounds");

        assert_eq!(
            rank_zero,
            Array::new(vec![221], rank_zero.shape().clone()).expect("hand verified")
        );

        let mut rank_zero_in_place = array.view();
        rank_zero_in_place
            .restrict(DimIndex(0), 4)
            .and_then(|array| array.restrict(DimIndex(1), 2))
            .and_then(|array| array.restrict(DimIndex(2), 1))
            .map(|array| array.drop_unit_extents())
            .expect("all in bounds");

        assert_eq!(rank_zero, rank_zero_in_place);
        assert_eq!(rank_zero.shape(), rank_zero_in_place.shape());
        assert_eq!(rank_zero.layout(), rank_zero_in_place.layout());
    }

    #[test]
    fn permuted_round_trip() {
        let shape = Shape::new(DynDim::from_array([2, 2, 3]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let order = DimOrder::new(DynDim::from_array([1, 2, 0])).expect("hand verified");

        let permuted = array.view().permuted(&order).expect("same rank");
        let restored = permuted
            .permuted(&order.inverse())
            .expect("same rank");

        assert_eq!(
            permuted,
            Array::new(
                vec![0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11],
                Shape::new(DynDim::from_array([2, 3, 2]))
            )
            .expect("hand verified")
        );
        assert_eq!(restored, array);

        let mut permuted_in_place = array.view();
        permuted_in_place
            .permute(&order)
            .expect("same rank");
        let mut restored_in_place = permuted_in_place.view();
        restored_in_place
            .permute(&order.inverse())
            .expect("same rank");

        assert_eq!(permuted, permuted_in_place);
        assert_eq!(restored, restored_in_place);
    }

    #[test]
    fn element_access_is_complete() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        for (expected, index) in array
            .shape()
            .indices_lexicographic()
            .unwrap()
            .enumerate()
            .map(|(expected, index)| (expected as i32, index))
        {
            assert_eq!(array[&index], expected);
            assert_eq!(array.get(&index), Some(&expected));
            assert_eq!(array.get_mut(&index).as_deref(), Some(&expected));
            // SAFETY: indices generated by the array's shape are always valid.
            assert_eq!(unsafe { *array.get_unchecked(&index) }, expected);
            assert_eq!(unsafe { *array.get_unchecked_mut(&index) }, expected);
        }
    }

    #[test]
    fn element_mutation_is_complete() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let indices = array
            .shape()
            .indices_lexicographic()
            .expect("hand verified")
            .collect::<Vec<ArrayIndex<DynDim<usize>>>>();

        for index in &indices {
            array[index] += 1;
        }
        for index in &indices {
            *array.get_mut(index).expect("in bounds") += 10;
        }
        for index in &indices {
            // SAFETY: indices generated by the array's shape are always valid.
            unsafe { *array.get_unchecked_mut(index) += 100 };
        }

        for index in &indices {
            let offset = array.layout().linear_unvalidated(index) as i32;

            assert_eq!(array[index], offset + 111);
        }
    }

    #[test]
    fn partial_eq_across_types() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let array = ArrayOwned::from_linear_fn(shape.clone(), |i| i as i32).expect("hand verified");

        assert_eq!(array, array);
        assert_eq!(array, array.view());
        assert_eq!(array.view(), array);
        assert_eq!(array.view(), array.view());

        let static_array = array
            .clone()
            .try_into_static::<2>()
            .expect("same rank");

        assert_eq!(array, static_array);
        assert_eq!(static_array, array);
        assert_eq!(array, static_array.view());
        assert_eq!(array.view(), static_array);
        assert_eq!(static_array, array.view());
        assert_eq!(static_array.view(), array);
        assert_eq!(array.view(), static_array.view());
        assert_eq!(static_array.view(), array.view());

        let layout = Layout::column_major(shape.clone(), 0).expect("hand verified");
        let owned = Box::<[i32]>::from([0, 3, 1, 4, 2, 5]);
        let column_major = Array::from_parts(owned, layout).expect("hand verified");
        let row_major = array;

        assert_eq!(row_major, column_major);
        assert_eq!(column_major, row_major);
        assert_eq!(row_major, column_major.view());
        assert_eq!(row_major.view(), column_major);
        assert_eq!(column_major, row_major.view());
        assert_eq!(column_major.view(), row_major);
        assert_eq!(column_major.view(), row_major.view());
        assert_eq!(column_major.view(), row_major.view());
    }

    #[test]
    fn rank_conversions() {
        let shape = Shape::new(StaticDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let dynamic = array.clone().into_dyn();

        assert_eq!(dynamic, array);

        let recovered = dynamic.try_into_static::<3>().expect("same rank");

        assert_eq!(recovered, array);

        let wrong = array.clone().into_dyn().try_into_static::<2>();

        assert!(wrong.is_err());
        assert_eq!(
            array,
            wrong.expect_err("incompatible rank conversion must fail")
        );
    }

    #[test]
    fn lane_length() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 0, 0]));

        for dim in (0..array.rank()).map(DimIndex) {
            let lane = array
                .lane_at(dim, &index)
                .expect("every dimension has a lane through the origin");

            assert_eq!(lane.len(), array.shape().as_slice()[dim.0]);
        }
    }

    #[test]
    fn lanes_and_element_access_equivalence() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        for dim in (0..array.rank()).map(DimIndex) {
            for index in array
                .shape()
                .indices_lexicographic()
                .expect("hand verified")
            {
                let lane = array.lane_at(dim, &index).expect("in bounds");

                assert_eq!(lane.len(), array.shape().as_slice()[dim.0]);
                assert_eq!(lane.is_contiguous(), lane.as_slice().is_some());
                assert_eq!(
                    lane.is_contiguous(),
                    lane.iter().try_into_contiguous().is_some()
                );
                assert!(lane.get(lane.len()).is_none());
                assert_eq!(lane.to_vec(), lane.iter().copied().collect::<Vec<i32>>());

                let mut curr = index.clone();
                for position in 0..lane.len() {
                    curr.as_mut_slice()[dim.0] = position;
                    let expected = array.get(&curr).expect("on the lane");

                    assert_eq!(lane.get(position), Some(expected));
                    // SAFETY: generated by the array's lanes always valid.
                    assert_eq!(unsafe { lane.get_unchecked(position) }, expected);
                }
            }
        }
    }

    #[test]
    fn contiguous_lanes() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let (dim, lanes) = array
            .contiguous_lanes()
            .expect("row-major has stride 1");

        assert_eq!(dim, DimIndex(2));

        let mut elements = Vec::new();
        for lane in lanes {
            let slice = lane
                .as_slice()
                .expect("`contiguous_lanes` guarantees this");
            elements.extend_from_slice(slice);

            assert_eq!(slice.len(), 4);
        }

        assert_eq!(elements, (0..array.len() as i32).collect::<Vec<i32>>());
    }

    #[test]
    fn contiguous_lanes_degenerate_dimensions() {
        let shape = Shape::new(DynDim::from_array([1, 10, 1]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let (dim, lanes) = array
            .contiguous_lanes()
            .expect("row-major has stride 1");
        let collected = lanes.collect::<Vec<Lane<i32>>>();

        assert_eq!(array.layout.strides().as_slice(), &[10, 1, 1]);
        assert_eq!(dim, DimIndex(1));
        assert_eq!(collected.len(), 1);
        assert_eq!(
            collected[0].as_slice(),
            Some(
                (0..array.len() as i32)
                    .collect::<Vec<i32>>()
                    .as_slice()
            )
        );
    }

    #[test]
    fn lane_iterator_constructors() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let order = layout.lexicographic_order();
        let order_too_high = DimOrder::lexicographic(3).expect("DynDim");
        let mut short = vec![0_i32; 5];
        let mut exact = vec![0_i32; 6];
        let mut long = vec![0_i32; 100];

        assert!(Lanes::new(&short, layout.clone(), DimIndex(0), order.clone()).is_none());
        assert!(Lanes::new(&exact, layout.clone(), DimIndex(0), order.clone()).is_some());
        assert!(Lanes::new(&long, layout.clone(), DimIndex(0), order.clone()).is_some());
        assert!(Lanes::new(&exact, layout.clone(), DimIndex(2), order.clone()).is_none());
        assert!(Lanes::new(&exact, layout.clone(), DimIndex(0), order_too_high.clone()).is_none());
        assert!(LanesMut::new(&mut short, layout.clone(), DimIndex(0), order.clone()).is_none());
        assert!(LanesMut::new(&mut exact, layout.clone(), DimIndex(0), order.clone()).is_some());
        assert!(LanesMut::new(&mut long, layout.clone(), DimIndex(0), order.clone()).is_some());
        assert!(LanesMut::new(&mut exact, layout.clone(), DimIndex(2), order.clone()).is_none());
        assert!(
            LanesMut::new(
                &mut exact,
                layout.clone(),
                DimIndex(0),
                order_too_high.clone()
            )
            .is_none()
        );
    }

    #[test]
    fn mutable_lanes_disjoint() {
        for extents in [
            &[2, 3, 4][..],
            &[3, 1, 2][..],
            &[1, 10, 1][..],
            &[5][..],
            &[],
        ] {
            for dim in (0..extents.len()).map(DimIndex) {
                for lexicographic in [false, true] {
                    let shape = Shape::new(DynDim::from_slice(extents));
                    let mut array =
                        ArrayOwned::from_linear_fn(shape, |_| 0_i32).expect("hand verified");
                    let lanes = if lexicographic {
                        array
                            .lanes_lexicographic_mut(dim)
                            .expect("non-overlapping")
                            .collect::<Vec<LaneMut<i32>>>()
                    } else {
                        array
                            .lanes_memory_order_mut(dim)
                            .expect("non-overlapping")
                            .collect::<Vec<LaneMut<i32>>>()
                    };
                    for lane in lanes {
                        for elem in lane {
                            *elem += 1;
                        }
                    }

                    assert!(
                        array
                            .storage
                            .as_slice()
                            .iter()
                            .all(|elem| *elem == 1)
                    );
                }
            }
        }
    }

    #[test]
    fn mutable_lanes_strided() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let dim = DimIndex(0);
        let index = ArrayIndex::new(DynDim::from_array([0, 1]));
        let mut lane = array.lane_at_mut(dim, &index).expect("in bounds");
        *lane.get_mut(0).expect("in bounds") = 100;
        // SAFETY: lanes (rows here) contain 2 elements each, so this is
        // guaranteed to be in bounds.
        unsafe { *lane.get_unchecked_mut(1) = 200 }

        assert!(lane.get(2).is_none());
        assert!(!lane.is_contiguous());
        assert!(lane.into_mut_slice().is_none());
        assert_eq!(array.storage.as_slice(), &[0, 100, 2, 3, 200, 5]);
    }

    #[test]
    fn contiguous_mutable_lanes() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let dim = DimIndex(1);
        let index = ArrayIndex::new(DynDim::from_array([1, 0]));
        let lane = array.lane_at_mut(dim, &index).expect("in bounds");
        let slice = lane
            .into_mut_slice()
            .expect("contiguous in row-major");

        assert_eq!(slice, &[3, 4, 5]);

        slice.fill(0);

        assert_eq!(array.storage.iter().as_slice(), &[0, 1, 2, 0, 0, 0]);
    }

    #[test]
    fn mutable_lane_reborrow() {
        fn sum(lane: LaneMut<'_, i32>) -> i32 {
            lane.into_iter().map(|i| *i).sum()
        }

        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 0, 0]));

        for dim in (0..array.rank()).map(DimIndex) {
            let mut lane = array.lane_at_mut(dim, &index).expect("in bounds");
            let shared = lane.as_lane();
            let before = shared.iter().copied().collect::<Vec<i32>>();
            *lane.get_mut(0).expect("in bounds") += 1;

            assert_eq!(sum(lane.reborrow()), before.iter().sum::<i32>() + 1);

            *lane.get_mut(0).expect("in bounds") -= 1;

            let view = lane.into_lane();

            assert_eq!(view.to_vec(), before);
        }
    }

    #[test]
    fn lanes_traversal() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");

        for dim in (0..array.rank()).map(DimIndex) {
            let mut iter = array.lanes_memory_order(dim).expect("in bounds");
            let expected = iter
                .clone()
                .map(|lane| lane.to_vec())
                .collect::<Vec<Vec<i32>>>();

            assert_eq!(iter.size_hint(), (expected.len(), Some(expected.len())));

            let (mut front, mut back) = (0, expected.len());
            let mut seen = Vec::new();

            while front < back {
                seen.push((front, iter.next().expect("not exhausted").to_vec()));
                front += 1;
                if front < back {
                    back -= 1;
                    seen.push((back, iter.next_back().expect("not exhausted").to_vec()));
                }

                assert_eq!(iter.len(), back - front);
            }

            assert!(iter.next().is_none());
            assert!(iter.next_back().is_none());
            assert!(iter.next().is_none());
            for (number, lane) in seen {
                assert_eq!(lane, expected[number]);
            }
        }
    }

    #[test]
    fn strided_iter_delegation() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 0, 0]));

        for dim in (0..array.rank()).map(DimIndex) {
            let lane = array.lane_at(dim, &index).expect("in bounds");
            let naive = lane.iter().copied().collect::<Vec<i32>>();
            let last = *naive.last().expect("non-empty");
            let mut manual = Vec::new();
            lane.iter().for_each(|&e| manual.push(e));

            assert_eq!(lane.iter().count(), naive.len());
            assert_eq!(lane.iter().last(), naive.last());
            assert_eq!(lane.iter().size_hint(), (naive.len(), Some(naive.len())));
            assert_eq!(
                lane.iter().fold(1, |acc, &e| acc + e) - 1,
                naive.iter().sum()
            );
            assert_eq!(
                lane.iter().rfold(1, |acc, &e| acc + e) - 1,
                naive.iter().sum()
            );
            assert_eq!(
                lane.iter().rev().copied().collect::<Vec<i32>>(),
                naive.iter().rev().copied().collect::<Vec<i32>>()
            );
            for n in 0..naive.len() {
                assert_eq!(lane.iter().nth(n), naive.get(n));
                assert_eq!(lane.iter().nth_back(n), naive.get(naive.len() - n - 1));
            }
            assert_eq!(lane.iter().find(|&&e| e == last), Some(&last));
            assert_eq!(lane.iter().rfind(|&&e| e == last), Some(&last));
            assert_eq!(lane.iter().position(|&e| e == last), Some(naive.len() - 1));
            assert_eq!(
                lane.iter()
                    .find_map(|&e| (e == last).then_some(e)),
                Some(last),
            );
            assert!(lane.iter().any(|&e| e == last));
            assert!(lane.iter().all(|&e| e <= last));
            assert_eq!(manual, naive);
        }
    }

    #[test]
    fn lane_element_iterators() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 2, 3]));
        let geometry = layout
            .lane_at(DimIndex(0), &index)
            .expect("in bounds");

        assert_eq!(
            (geometry.offset(), geometry.stride(), geometry.len()),
            (11, 12, 2)
        );

        let mut short = vec![0_i32; 23];
        let mut exact = vec![0_i32; 24];
        let mut long = vec![0_i32; 100];

        assert!(LaneElemStrided::new(&short, geometry).is_none());
        assert!(LaneElemStrided::new(&exact, geometry).is_some());
        assert!(LaneElemStrided::new(&long, geometry).is_some());
        assert!(LaneElemStridedMut::new(&mut short, geometry).is_none());
        assert!(LaneElemStridedMut::new(&mut exact, geometry).is_some());
        assert!(LaneElemStridedMut::new(&mut long, geometry).is_some());

        let iter = LaneElemStrided::new(&exact, geometry).expect("fits");

        assert_eq!(iter.stride(), 12);
        assert_eq!(iter.len(), iter.count());
    }

    #[test]
    fn numpy_like_broadcasts() {
        let shape = Shape::new(DynDim::from_array([3, 2]));
        let broadcast = Strides::new(DynDim::from_array([0, 1]));
        let layout = Layout::new(shape, broadcast, 0).expect("hand verified");

        assert_eq!(layout.max_offset(), 1);
        assert_eq!(layout.len(), 6);
        assert!(!layout.is_non_overlapping());

        let storage = vec![10_i32, 20_i32].into_boxed_slice();
        let mut array = Array::from_parts(storage, layout).expect("hand verified");

        for pseudo_row in 0..array.shape().get(DimIndex(0)).unwrap() {
            assert_eq!(array[[pseudo_row, 0]], 10);
            assert_eq!(array[[pseudo_row, 1]], 20);
        }
        for (number, lane) in array
            .lanes_memory_order(DimIndex(0))
            .expect("in bounds")
            .enumerate()
        {
            assert_eq!(lane.to_vec(), vec![10 * (number as i32 + 1); 3]);
        }
        for dim in (0..array.rank()).map(DimIndex) {
            assert!(array.lanes_memory_order_mut(dim).is_none());
            assert!(array.lanes_memory_order_mut(dim).is_none());
        }

        let index = ArrayIndex::new(DynDim::from_array([0, 0]));
        let mut lane = array
            .lane_at_mut(DimIndex(1), &index)
            .expect("in bounds");
        *lane.get_mut(0).expect("in bounds") = 15;
        *lane.get_mut(1).expect("in bounds") = 25;

        for pseudo_row in 0..array.shape().get(DimIndex(0)).unwrap() {
            assert_eq!(array[[pseudo_row, 0]], 15);
            assert_eq!(array[[pseudo_row, 1]], 25);
        }
        for (number, lane) in array
            .lanes_memory_order(DimIndex(0))
            .expect("in bounds")
            .enumerate()
        {
            assert_eq!(lane.to_vec(), vec![10 * (number as i32 + 1) + 5; 3]);
        }
        for dim in (0..array.rank()).map(DimIndex) {
            assert!(array.lanes_memory_order_mut(dim).is_none());
            assert!(array.lanes_memory_order_mut(dim).is_none());
        }
    }

    #[test]
    fn mutable_lanes_with_threads() {
        let shape = Shape::new(DynDim::from_array([4, 5]));
        let mut array = ArrayOwned::from_linear_fn(shape, |i| i as i32).expect("hand verified");
        let lanes = array
            .lanes_memory_order_mut(DimIndex(1))
            .expect("in bounds");

        std::thread::scope(|scope| {
            for (number, lane) in lanes.enumerate() {
                scope.spawn(move || {
                    for elem in lane {
                        *elem = number as i32;
                    }
                });
            }
        });

        for row in 0..array.shape().get(DimIndex(0)).unwrap() {
            for column in 0..array.shape().get(DimIndex(1)).unwrap() {
                assert_eq!(array[[row, column]], row as i32);
            }
        }
    }

    #[test]
    fn zero_sized_elements() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let mut array = ArrayOwned::from_linear_fn(shape, |_| ()).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([1, 1, 1]));

        for dim in (0..array.rank()).map(DimIndex) {
            let lane = array.lane_at(dim, &index).expect("in bounds");

            assert_eq!(lane.iter().count(), lane.len());
            assert_eq!(lane.get(lane.len() - 1), Some(&()));
            // SAFETY: index < len guarantees being in bounds.
            assert_eq!(unsafe { lane.get_unchecked(lane.len() - 1) }, &());

            let lanes = array
                .lanes_memory_order_mut(dim)
                .expect("non-overlapping")
                .collect::<Vec<LaneMut<'_, ()>>>();
            for lane in lanes {
                for elem in lane {
                    *elem = ();
                }
            }

            for index in array
                .shape()
                .indices_lexicographic()
                .expect("hand verified")
            {
                assert_eq!(array[&index], ());
                assert_eq!(array.get(&index), Some(&()));
                // SAFETY: indices generated by the array's shape are guaranteed
                // to be within bounds.
                assert_eq!(unsafe { array.get_unchecked(&index) }, &());
            }
        }
    }
}
