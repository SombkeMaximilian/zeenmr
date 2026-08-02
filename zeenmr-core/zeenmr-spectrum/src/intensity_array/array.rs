use crate::intensity_array::{
    ArrayIndex, Dimension, DynDim, Layout, Shape, StaticDim, Storage, StorageMut, StorageOwned,
};
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

/// Array borrowing its storage immutably.
pub type ArrayView<'s, T, D> = Array<&'s [T], D>;

/// Array borrowing its storage mutably.
pub type ArrayViewMut<'s, T, D> = Array<&'s mut [T], D>;

/// Array owning its storage.
pub type ArrayOwned<T, D> = Array<Box<[T]>, D>;

/// Array using clone-on-write storage.
pub type ArrayCow<'s, T, D> = Array<Cow<'s, [T]>, D>;

/// Array using referencing counting.
pub type ArrayRc<T, D> = Array<Rc<[T]>, D>;

/// Array using atomic reference counting.
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
#[derive(Debug)]
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

impl<S, D> Array<S, D>
where
    D: Dimension,
{
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
