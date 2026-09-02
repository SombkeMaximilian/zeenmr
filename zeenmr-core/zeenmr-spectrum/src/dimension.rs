//! Types and traits for multidimensional quantities.

use std::borrow::Cow;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};

/// Index into a multidimensional quantity.
///
/// Such a quantity generally has `N` dimensions. This type encapsulates the
/// index `i` into such a quantity with `0 <= i < N`.
///
/// This type has the same size and alignment as `usize`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
pub struct DimIndex(pub usize);

/// Abstraction for multidimensional quantities.
pub trait Dimension: Clone + Send + Sync {
    /// Element type of the quantity.
    type Elem: Clone + Send + Sync;

    /// Compile time constant rank, if available.
    const RANK: Option<usize>;

    /// Create a multidimensional quantity of the provided rank by filling it
    /// using `f`.
    ///
    /// `f` is called exactly once per dimension, with `0, 1, ..., rank - 1` in
    /// ascending order.
    ///
    /// Returns `None` if this type can't represent the rank.
    fn from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Self::Elem;

    /// Create a multidimensional quantity of the provided rank by filling it
    /// using `f`.
    ///
    /// `f` is called with `0, 1, ..., rank - 1` in ascending order, and is not
    /// called again once it has returned `None`.
    ///
    /// Returns `None` if this type can't represent the rank, or if `f` returns
    /// `None`.
    fn try_from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Option<Self::Elem>;

    /// Returns the equivalent instance of `self`, or `None` if this type cannot
    /// represent `other`'s rank.
    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension<Elem = Self::Elem>;

    /// Creates a multidimensional quantity from an iterator.
    ///
    /// Returns `None` if this type can't represent a rank equal to the
    /// iterator's length.
    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Elem>,
        I::IntoIter: ExactSizeIterator;

    /// Returns the rank of `self`.
    fn rank(&self) -> usize;

    /// Returns a reference to the dimension element at `index`.
    fn get(&self, index: DimIndex) -> Option<&Self::Elem>;

    /// Returns a mutable reference to the dimension element at `index`.
    fn get_mut(&mut self, index: DimIndex) -> Option<&mut Self::Elem>;

    /// Returns a slice containing all dimensions of this quantity.
    fn as_slice(&self) -> &[Self::Elem];

    /// Returns a mutable slice containing all dimensions of this quantity.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

/// Conversion into a [`Dimension`].
pub trait IntoDimension {
    /// Dimension type this input can be converted into.
    type Dim: Dimension;

    /// Creates a dimension type from `self`.
    fn into_dim(self) -> Self::Dim;
}

impl<T, const N: usize> IntoDimension for StaticDim<T, N>
where
    T: Clone + Send + Sync,
{
    type Dim = Self;

    fn into_dim(self) -> Self::Dim {
        self
    }
}

impl<T, const N: usize> IntoDimension for [T; N]
where
    T: Clone + Send + Sync,
{
    type Dim = StaticDim<T, N>;

    fn into_dim(self) -> Self::Dim {
        StaticDim::from_array(self)
    }
}

impl<T> IntoDimension for DynDim<T>
where
    T: Clone + Send + Sync,
{
    type Dim = Self;

    fn into_dim(self) -> Self::Dim {
        self
    }
}

impl<T> IntoDimension for &[T]
where
    T: Clone + Send + Sync,
{
    type Dim = DynDim<T>;

    fn into_dim(self) -> Self::Dim {
        DynDim::from_slice(self)
    }
}

impl<T> IntoDimension for &mut [T]
where
    T: Clone + Send + Sync,
{
    type Dim = DynDim<T>;

    fn into_dim(self) -> Self::Dim {
        DynDim::from_slice(self)
    }
}

impl<T> IntoDimension for Vec<T>
where
    T: Clone + Send + Sync,
{
    type Dim = DynDim<T>;

    fn into_dim(self) -> Self::Dim {
        DynDim::from_vec(self)
    }
}

impl<T> IntoDimension for Box<[T]>
where
    T: Clone + Send + Sync,
{
    type Dim = DynDim<T>;

    fn into_dim(self) -> Self::Dim {
        DynDim::from_vec(self.into_vec())
    }
}

impl<T> IntoDimension for Cow<'_, [T]>
where
    T: Clone + Send + Sync,
{
    type Dim = DynDim<T>;

    fn into_dim(self) -> Self::Dim {
        DynDim::from_vec(self.into_owned())
    }
}

/// Asserts that the two dimension types have compatible ranks.
///
/// The only case where this can fail is `StaticDim<T, N>` and `StaticDim<T, M>`
/// where `N != M`, since either of the two types being `DynDim<T>` makes them
/// always compatible. Note that this does *not* compare the actual runtime
/// ranks and only serves to elide runtime checks for `StaticDim`.
pub(crate) const fn assert_rank_compatible<D1, D2>()
where
    D1: Dimension,
    D2: Dimension,
{
    assert!(match (D1::RANK, D2::RANK) {
        (Some(d1_rank), Some(d2_rank)) => d1_rank == d2_rank,
        // at least one of them is dynamic, so it can always represent the other
        _ => true,
    })
}

/// Multidimensional quantity with a size determined at compile-time.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct StaticDim<T, const N: usize>([T; N]);

impl<T, const N: usize> From<[T; N]> for StaticDim<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> IntoIterator for StaticDim<T, N> {
    type Item = T;
    type IntoIter = core::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, const N: usize> Dimension for StaticDim<T, N>
where
    T: Clone + Send + Sync,
{
    type Elem = T;

    const RANK: Option<usize> = Some(N);

    fn from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Self::Elem,
    {
        if rank == N {
            Some(Self(core::array::from_fn(f)))
        } else {
            None
        }
    }

    // for the future: should be replaced by `array::try_from_fn` whenever that
    // is stabilized (https://github.com/rust-lang/rust/issues/89379)
    fn try_from_fn<F>(rank: usize, mut f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Option<Self::Elem>,
    {
        if rank != N {
            return None;
        }

        let mut elem = [const { MaybeUninit::<T>::uninit() }; N];
        let mut guard = Guard {
            array_mut: &mut elem,
            initialized: 0,
        };
        while guard.initialized < N {
            // SAFETY: the loop condition guarantees this is in bounds.
            unsafe { guard.push_unchecked(f(guard.initialized)?) };
        }
        mem::forget(guard);

        // SAFETY: all elements of the array were successfully initialized.
        Some(Self(elem.map(|e| unsafe { e.assume_init() })))
    }

    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension<Elem = Self::Elem>,
    {
        const { assert_rank_compatible::<Self, D>() };

        if other.rank() == N {
            let slice = other.as_slice();

            Self::from_fn(other.rank(), |dim| slice[dim].clone())
        } else {
            None
        }
    }

    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Elem>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();

        if iter.len() == N {
            Self::try_from_fn(N, |_| iter.next())
        } else {
            None
        }
    }

    fn rank(&self) -> usize {
        N
    }

    fn get(&self, index: DimIndex) -> Option<&Self::Elem> {
        self.0.get(index.0)
    }

    fn get_mut(&mut self, index: DimIndex) -> Option<&mut Self::Elem> {
        self.0.get_mut(index.0)
    }

    fn as_slice(&self) -> &[Self::Elem] {
        self.0.as_ref()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.0.as_mut()
    }
}

impl<T, const N: usize> StaticDim<T, N> {
    /// Creates a new static dimension from an array.
    pub fn from_array(value: [T; N]) -> Self {
        Self(value)
    }

    /// Returns the underlying array.
    pub fn into_array(self) -> [T; N] {
        self.0
    }
}

/// Multidimensional quantity with a size determined at runtime.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct DynDim<T>(Vec<T>);

impl<T, const N: usize> From<[T; N]> for DynDim<T>
where
    T: Clone,
{
    fn from(value: [T; N]) -> Self {
        Self::from_array(value)
    }
}

impl<T> From<&[T]> for DynDim<T>
where
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        Self::from_slice(value)
    }
}

impl<T> From<Vec<T>> for DynDim<T>
where
    T: Clone,
{
    fn from(value: Vec<T>) -> Self {
        Self::from_vec(value)
    }
}

impl<T> FromIterator<T> for DynDim<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<T> Dimension for DynDim<T>
where
    T: Clone + Send + Sync,
{
    type Elem = T;

    const RANK: Option<usize> = None;

    fn from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Self::Elem,
    {
        Some((0..rank).map(f).collect())
    }

    fn try_from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Option<Self::Elem>,
    {
        (0..rank).map(f).collect()
    }

    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension<Elem = Self::Elem>,
    {
        Some(Self::from_slice(other.as_slice()))
    }

    fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Elem>,
        I::IntoIter: ExactSizeIterator,
    {
        Some(iter.into_iter().collect())
    }

    fn rank(&self) -> usize {
        self.as_slice().len()
    }

    fn get(&self, index: DimIndex) -> Option<&Self::Elem> {
        self.0.get(index.0)
    }

    fn get_mut(&mut self, index: DimIndex) -> Option<&mut Self::Elem> {
        self.0.get_mut(index.0)
    }

    fn as_slice(&self) -> &[Self::Elem] {
        self.0.deref()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.0.deref_mut()
    }
}

impl<T> DynDim<T> {
    /// Creates a representation from a `Vec<T>`.
    pub fn from_vec(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T> DynDim<T>
where
    T: Clone,
{
    /// Creates a new dynamic dimensional quantity from an array.
    pub fn from_array<const N: usize>(value: [T; N]) -> Self {
        Self(value.into())
    }

    /// Creates a new dynamic dimensional quantity from a slice.
    ///
    /// Prefer [`DynDim::from_vec`] or the [`FromIterator`] implementation for
    /// passing owned values.
    pub fn from_slice<S>(value: S) -> Self
    where
        S: AsRef<[T]>,
    {
        Self(value.as_ref().to_vec())
    }
}

/// Panic guard for array initialization.
///
/// Inspired by the standard library's `array::try_from_fn` implementation.
///
/// # Invariants
///
/// All write accesses to this structure are unsafe and must maintain a correct
/// count of `initialized` elements.
///
/// Call [`mem::forget`] on the guard once initialization completes.
struct Guard<'a, T> {
    /// Array to be initialized.
    array_mut: &'a mut [MaybeUninit<T>],
    /// Number of items that are correctly initialized.
    initialized: usize,
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        debug_assert!(self.initialized <= self.array_mut.len());
        let initialized = &mut self.array_mut[..self.initialized];
        if !initialized.is_empty() {
            // SAFETY: `MaybeUninit<T>` and `T` share a layout, and the type
            // invariant guarantees these are initialized.
            unsafe { core::ptr::drop_in_place(initialized as *mut [MaybeUninit<T>] as *mut [T]) };
        }
    }
}

impl<T> Guard<'_, T> {
    /// Initializes the next element as `value`.
    ///
    /// # Safety
    ///
    /// Must not be called more times than there are elements in the array.
    unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.initialized < self.array_mut.len());

        // SAFETY: as long as `initialized` is correct and this method is not
        // called more times than the capacity of the array, every write will
        // be in bounds.
        unsafe {
            self.array_mut
                .get_unchecked_mut(self.initialized)
                .write(value);
            self.initialized = self.initialized.unchecked_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    /// Crosses the inline boundary.
    const RANKS: [usize; 6] = [0, 1, 2, 3, 4, 9];

    #[test]
    fn dim_index_size_alignment() {
        const _: () = {
            assert!(size_of::<DimIndex>() == size_of::<usize>());
            assert!(align_of::<DimIndex>() == align_of::<usize>());
        };
    }

    #[test]
    fn rank_compatible() {
        const {
            assert_rank_compatible::<DynDim<i32>, DynDim<i32>>();
            assert_rank_compatible::<DynDim<i32>, DynDim<u64>>();
            assert_rank_compatible::<DynDim<i32>, StaticDim<u64, 2>>();
            assert_rank_compatible::<StaticDim<u64, 2>, DynDim<i32>>();
            assert_rank_compatible::<StaticDim<i32, 2>, StaticDim<i32, 2>>();
            assert_rank_compatible::<StaticDim<i32, 2>, StaticDim<u64, 2>>();
        }
    }

    #[test]
    fn static_dynamic_bridge() {
        let dim0 = DynDim::from_array([]);
        let dim1 = DynDim::from_array([1]);
        let dim2 = DynDim::from_array([1, 2]);
        let dim3 = DynDim::from_array([1, 2, 3]);
        let dim4 = DynDim::from_array([1, 2, 3, 4]);
        let dim5 = DynDim::from_array([1, 2, 3, 4, 5]);

        assert!(StaticDim::<i32, 0>::from_dimension(&dim0).is_some());
        assert!(StaticDim::<i32, 1>::from_dimension(&dim1).is_some());
        assert!(StaticDim::<i32, 2>::from_dimension(&dim2).is_some());
        assert!(StaticDim::<i32, 3>::from_dimension(&dim3).is_some());
        assert!(StaticDim::<i32, 4>::from_dimension(&dim4).is_some());
        assert!(StaticDim::<i32, 5>::from_dimension(&dim5).is_some());
    }

    #[test]
    fn conversion() {
        let stat = StaticDim::from_array([
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        ]);
        let converted = DynDim::from_dimension(&stat).expect("DynDim can represent any rank");
        let recovered = StaticDim::<NonZeroUsize, 3>::from_dimension(&converted)
            .expect("round trip must never fail");

        assert_eq!(stat.as_slice(), converted.as_slice());
        assert_eq!(stat, recovered);

        let dynamic = DynDim::from_array([
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        ]);
        let converted = StaticDim::<NonZeroUsize, 3>::from_dimension(&dynamic).expect("same rank");
        let recovered = DynDim::from_dimension(&converted).expect("DynDim can represent any rank");

        assert_eq!(converted.as_slice(), dynamic.as_slice());
        assert_eq!(recovered, dynamic);
        assert!(StaticDim::<NonZeroUsize, 2>::from_dimension(&dynamic).is_none());
        assert!(StaticDim::<NonZeroUsize, 4>::from_dimension(&dynamic).is_none());
    }

    #[test]
    fn from_fn_respects_rank() {
        for rank in RANKS {
            let dynamic = DynDim::from_fn(rank, |dim| dim).expect("any rank");

            assert_eq!(dynamic.rank(), rank);
            assert_eq!(dynamic.as_slice(), (0..rank).collect::<Vec<_>>());
        }

        assert!(StaticDim::<usize, 3>::from_fn(2, |dim| dim).is_none());
        assert!(StaticDim::<usize, 3>::from_fn(3, |dim| dim).is_some());
        assert!(StaticDim::<usize, 3>::from_fn(4, |dim| dim).is_none());
    }

    #[test]
    fn from_fn_is_ascending() {
        for rank in RANKS {
            let mut seen = Vec::new();
            let _ = DynDim::from_fn(rank, |dim| {
                seen.push(dim);

                dim
            })
            .expect("any rank");

            assert_eq!(seen, (0..rank).collect::<Vec<usize>>());
        }
    }

    #[test]
    fn try_from_fn_fail() {
        for rank in RANKS {
            for fail_at in 0..rank {
                let mut calls = 0;
                let result = DynDim::try_from_fn(rank, |dim| {
                    calls += 1;

                    if dim != fail_at { Some(dim) } else { None }
                });

                assert!(result.is_none());
                assert_eq!(calls, fail_at + 1);
            }
        }
    }

    #[test]
    fn slice_mutation() {
        for rank in RANKS {
            let mut dynamic = DynDim::from_fn(rank, |_| 0_usize).expect("any rank");
            dynamic
                .as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(dim, slot)| *slot = dim);

            assert_eq!(dynamic.as_slice(), (0..rank).collect::<Vec<_>>());
        }
    }
}
