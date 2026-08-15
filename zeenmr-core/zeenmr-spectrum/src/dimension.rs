//! Types and traits for multidimensional quantities.

use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};

/// Maximum number of non-heap dimensions in the dynamic case.
///
/// Limited to 3D to keep the most common cases as lean as possible.
const MAX_INLINE_RANK: usize = 3;

/// Index into a multidimensional quantity.
///
/// Such a quantity generally has `N` dimensions. This type encapsulates the
/// index `i` into such a quantity with `0 <= i < N`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DimIndex(pub usize);

/// Abstraction for multidimensional quantities.
pub trait Dimension: Clone + Eq + Send + Sync + Sized {
    /// Element type of the quantity.
    type Elem: Clone;

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

    /// Returns a slice containing all dimensions of this quantity.
    fn as_slice(&self) -> &[Self::Elem];

    /// Returns a mutable slice containing all dimensions of this quantity.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];

    /// Returns the rank of `self`.
    fn rank(&self) -> usize;
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

impl<T, const N: usize> Dimension for StaticDim<T, N>
where
    T: Clone + Eq + Send + Sync,
{
    type Elem = T;

    const RANK: Option<usize> = Some(N);

    #[inline]
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

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        self.0.as_ref()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.0.as_mut()
    }

    #[inline]
    fn rank(&self) -> usize {
        N
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

/// Multidimensional quantity representation with a size determined at runtime.
///
/// # Invariants
///
/// In the `Stack` variant, `rank <= MAX_INLINE_RANK`, and every element of
/// `storage` at a position `i < rank` is initialized. Positions `i >= rank`
/// must never be read.
enum DynDimInner<T> {
    /// On the stack for up to [`MAX_INLINE_RANK`].
    Stack {
        /// Backing storage.
        storage: [MaybeUninit<T>; MAX_INLINE_RANK],
        /// Rank *and* number of initialized elements.
        rank: u8,
    },
    /// On the heap for higher dimensions.
    Heap(Vec<T>),
}

impl<T> Drop for DynDimInner<T> {
    fn drop(&mut self) {
        if let Self::Stack { storage, rank } = self {
            let initialized = &mut storage[..*rank as usize];
            // SAFETY: `MaybeUninit<T>` and `T` share a layout, and the type
            // invariant guarantees these are initialized.
            unsafe { core::ptr::drop_in_place(initialized as *mut [MaybeUninit<T>] as *mut [T]) };
        }
    }
}

impl<T> Deref for DynDimInner<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Stack { storage, rank } => {
                // SAFETY: `MaybeUninit<T>` and `T` share a layout, and the type
                // invariant guarantees these are initialized.
                unsafe {
                    let initialized = storage.get_unchecked(..*rank as usize);

                    &*(initialized as *const [MaybeUninit<T>] as *const [T])
                }
            }
            Self::Heap(storage) => storage,
        }
    }
}

impl<T> DerefMut for DynDimInner<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Stack { storage, rank } => {
                // SAFETY: `MaybeUninit<T>` and `T` share a layout, and the type
                // invariant guarantees these are initialized.
                unsafe {
                    let initialized = storage.get_unchecked_mut(..*rank as usize);

                    &mut *(initialized as *mut [MaybeUninit<T>] as *mut [T])
                }
            }
            Self::Heap(storage) => storage,
        }
    }
}

impl<T> Clone for DynDimInner<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::from_slice(&**self)
    }
}

impl<T> PartialEq for DynDimInner<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T> Eq for DynDimInner<T> where T: Eq {}

impl<T> std::hash::Hash for DynDimInner<T>
where
    T: std::hash::Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.deref().hash(state)
    }
}

impl<T> std::fmt::Debug for DynDimInner<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T> DynDimInner<T> {
    /// Creates a new dynamic dimensional quantity representation of the given
    /// rank by filling it using `f`.
    // for the future: should be replaced by `array::try_from_fn` whenever that
    // is stabilized (https://github.com/rust-lang/rust/issues/89379)
    fn try_from_fn<F>(rank: usize, mut f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Option<T>,
    {
        if rank <= MAX_INLINE_RANK {
            let mut storage = [const { MaybeUninit::<T>::uninit() }; MAX_INLINE_RANK];
            let mut guard = Guard {
                array_mut: &mut storage,
                initialized: 0,
            };
            while guard.initialized < rank {
                // SAFETY: the loop condition guarantees this is in bounds.
                unsafe { guard.push_unchecked(f(guard.initialized)?) };
            }
            mem::forget(guard);

            Some(Self::Stack {
                storage,
                rank: rank as u8,
            })
        } else {
            (0..rank)
                .map(f)
                .collect::<Option<Vec<T>>>()
                .map(Self::Heap)
        }
    }
}

impl<T> DynDimInner<T>
where
    T: Clone,
{
    /// Creates a new dynamic dimensional quantity representation from an array.
    fn from_array<const N: usize>(value: [T; N]) -> Self {
        if N <= MAX_INLINE_RANK {
            let mut storage = [const { MaybeUninit::<T>::uninit() }; MAX_INLINE_RANK];
            for (slot, elem) in storage.iter_mut().zip(value) {
                slot.write(elem);
            }

            Self::Stack {
                storage,
                rank: N as u8,
            }
        } else {
            Self::Heap(value.to_vec())
        }
    }

    /// Creates a new dynamic dimensional quantity representation from a slice.
    // for the future: inline branch should be replaced by `array::try_from_fn`
    // whenever that is stabilized
    // (https://github.com/rust-lang/rust/issues/89379)
    fn from_slice<S>(value: S) -> Self
    where
        S: AsRef<[T]> + Into<Vec<T>>,
    {
        let rank = value.as_ref().len();

        if rank <= MAX_INLINE_RANK {
            let value = value.as_ref();
            let mut storage = [const { MaybeUninit::<T>::uninit() }; MAX_INLINE_RANK];
            let mut guard = Guard {
                array_mut: &mut storage,
                initialized: 0,
            };
            while guard.initialized < rank {
                // SAFETY: the loop condition guarantees this is in bounds.
                unsafe { guard.push_unchecked(value[guard.initialized].clone()) };
            }
            mem::forget(guard);

            Self::Stack {
                storage,
                rank: rank as u8,
            }
        } else {
            Self::Heap(value.into())
        }
    }
}

/// Multidimensional quantity with a size determined at runtime.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct DynDim<T>(DynDimInner<T>);

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
        Self::from_slice(value)
    }
}

impl<T> Dimension for DynDim<T>
where
    T: Clone + Eq + Send + Sync,
{
    type Elem = T;

    const RANK: Option<usize> = None;

    fn from_fn<F>(rank: usize, mut f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Self::Elem,
    {
        Self::try_from_fn(rank, |dim| Some(f(dim)))
    }

    fn try_from_fn<F>(rank: usize, f: F) -> Option<Self>
    where
        F: FnMut(usize) -> Option<Self::Elem>,
    {
        Some(Self(DynDimInner::try_from_fn(rank, f)?))
    }

    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension<Elem = Self::Elem>,
    {
        Some(Self::from_slice(other.as_slice()))
    }

    fn as_slice(&self) -> &[Self::Elem] {
        self.0.deref()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.0.deref_mut()
    }

    fn rank(&self) -> usize {
        self.as_slice().len()
    }
}

impl<T> DynDim<T>
where
    T: Clone,
{
    /// Creates a new dynamic dimensional quantity from an array.
    pub fn from_array<const N: usize>(value: [T; N]) -> Self {
        Self(DynDimInner::from_array(value))
    }

    /// Creates a new dynamic dimensional quantity from a slice.
    pub fn from_slice<S>(value: S) -> Self
    where
        S: AsRef<[T]> + Into<Vec<T>>,
    {
        Self(DynDimInner::from_slice(value))
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Crosses the inline boundary.
    const RANKS: [usize; 6] = [0, 1, 2, 3, 4, 9];

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
    fn dynamic_stack_and_heap() {
        for rank in RANKS {
            let dynamic =
                DynDim::from_fn(rank, |_| 0_usize).expect("DynDim can represent any rank");
            let stack = matches!(dynamic.0, DynDimInner::Stack { .. });

            assert_eq!(stack, rank <= MAX_INLINE_RANK);
            assert_eq!(dynamic.clone(), dynamic);
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

    #[test]
    fn equality_and_hash() {
        let hash = |d: &DynDim<usize>| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(d, &mut hasher);

            std::hash::Hasher::finish(&hasher)
        };

        for rank in RANKS {
            let left = DynDim::from_fn(rank, |dim| dim).expect("any rank");
            let right = DynDim::from_fn(rank, |dim| dim).expect("any rank");

            assert_eq!(left, right);
            assert_eq!(hash(&left), hash(&right));
        }

        assert_ne!(DynDim::from_array([1, 2]), DynDim::from_array([1, 2, 3]));
    }

    /// Increment the shared counter when dropped.
    #[derive(Clone)]
    struct DropCount<'a>(&'a AtomicUsize);

    impl Drop for DropCount<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl PartialEq for DropCount<'_> {
        fn eq(&self, other: &Self) -> bool {
            std::ptr::eq(self.0, other.0)
        }
    }

    impl Eq for DropCount<'_> {}

    /// Payload marker for panics we started.
    struct Deliberate;

    /// Starts an unwind that [`expect_unwind`] recognizes.
    fn unwind_deliberately() -> ! {
        std::panic::resume_unwind(Box::new(Deliberate))
    }

    /// Runs `f`, asserting that it unwinds.
    #[track_caller]
    fn expect_unwind<F, R>(f: F)
    where
        F: FnOnce() -> R,
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(_) => panic!("expected an unwind"),
            Err(payload) if payload.is::<Deliberate>() => {}
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn inline_elements_drop() {
        for rank in RANKS {
            let drops = AtomicUsize::new(0);
            let inner = DynDimInner::try_from_fn(rank, |_| Some(DropCount(&drops)))
                .expect("DynDim can represent any rank");

            assert_eq!(drops.load(Ordering::Relaxed), 0);
            drop(inner);
            assert_eq!(drops.load(Ordering::Relaxed), rank);
        }
    }

    #[test]
    fn partial_initialization_drop() {
        for rank in RANKS {
            for fail_at in 0..rank {
                let drops = AtomicUsize::new(0);
                let result = DynDimInner::try_from_fn(rank, |dim| {
                    if dim != fail_at {
                        Some(DropCount(&drops))
                    } else {
                        None
                    }
                });

                assert!(result.is_none());
                assert_eq!(drops.load(Ordering::Relaxed), fail_at);
            }
        }
        const N: usize = 3;
        for fail_at in 0..N {
            let drops = AtomicUsize::new(0);
            let result = StaticDim::<DropCount, N>::try_from_fn(N, |dim| {
                if dim != fail_at {
                    Some(DropCount(&drops))
                } else {
                    None
                }
            });

            assert!(result.is_none());
            assert_eq!(drops.load(Ordering::Relaxed), fail_at);
        }
    }

    #[test]
    fn try_from_fn_unwind_drop() {
        for rank in RANKS {
            for panic_at in 0..rank {
                let drops = AtomicUsize::new(0);
                expect_unwind(|| {
                    DynDim::try_from_fn(rank, |dim| {
                        if dim == panic_at {
                            unwind_deliberately();
                        }

                        Some(DropCount(&drops))
                    })
                });

                assert_eq!(drops.load(Ordering::Relaxed), panic_at);
            }
        }
        const N: usize = 3;
        for panic_at in 0..N {
            let drops = AtomicUsize::new(0);
            expect_unwind(|| {
                StaticDim::<DropCount, N>::try_from_fn(N, |dim| {
                    if dim == panic_at {
                        unwind_deliberately();
                    }

                    Some(DropCount(&drops))
                })
            });

            assert_eq!(drops.load(Ordering::Relaxed), panic_at);
        }
    }

    /// Panics on the `panic_at`-th clone, and counts its own drops.
    struct PanicOnClone<'a> {
        /// Number of drops.
        drops: &'a AtomicUsize,
        /// Number of clones.
        clones: &'a AtomicUsize,
        /// Panics once `self.clones == self.panic_at`.
        panic_at: usize,
    }

    impl Clone for PanicOnClone<'_> {
        fn clone(&self) -> Self {
            let seen = self.clones.fetch_add(1, Ordering::Relaxed);
            if seen == self.panic_at {
                unwind_deliberately();
            }

            Self {
                drops: self.drops,
                clones: self.clones,
                panic_at: self.panic_at,
            }
        }
    }

    impl Drop for PanicOnClone<'_> {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn from_slice_unwind_drop() {
        for rank in RANKS {
            for panic_at in 0..rank {
                let (drops, clones) = (AtomicUsize::new(0), AtomicUsize::new(0));
                let source = (0..rank)
                    .map(|_| PanicOnClone {
                        drops: &drops,
                        clones: &clones,
                        panic_at,
                    })
                    .collect::<Vec<PanicOnClone>>();
                expect_unwind(|| DynDimInner::from_slice(source.as_slice()));

                assert_eq!(drops.load(Ordering::Relaxed), panic_at);
                assert_eq!(clones.load(Ordering::Relaxed), panic_at + 1);
            }
        }
    }
}
