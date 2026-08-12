use crate::intensity_array::iter::{Indices, LaneGeometries, LaneOffsets};
#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{ParIndices, ParLaneGeometries};
use std::ops::{Deref, DerefMut};

/// Maximum number of non-heap dimensions in the dynamic case.
///
/// Limited to 3D to keep the most common cases as lean as possible.
const MAX_INLINE_RANK: usize = 3;

/// Array dimension index.
///
/// An array generally has `N` dimensions. This type encapsulates the index `i`
/// of such a dimension with `0 <= i < N`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DimIndex(pub usize);

/// Abstraction for multidimensional quantities.
pub trait Dimension: Clone + Eq + Send + Sync {
    /// Compile time constant rank, if available.
    const RANK: Option<usize>;

    /// Returns a value of the given rank with all entries zero, or `None` if
    /// this type cannot represent it.
    fn zero(rank: usize) -> Option<Self>;

    /// Returns the equivalent instance of `self`, or `None` if this type cannot
    /// represent `other`'s rank.
    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension;

    /// Returns a slice containing all dimensions of this quantity.
    fn as_slice(&self) -> &[usize];

    /// Returns a mutable slice containing all dimensions of this quantity.
    fn as_mut_slice(&mut self) -> &mut [usize];

    /// Returns the rank of `self`.
    #[inline]
    fn rank(&self) -> usize {
        self.as_slice().len()
    }
}

/// Multidimensional quantity with a size determined at compile-time.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct StaticDim<const N: usize>([usize; N]);

impl<const N: usize> From<[usize; N]> for StaticDim<N> {
    fn from(value: [usize; N]) -> Self {
        Self(value)
    }
}

impl<const N: usize> Default for StaticDim<N> {
    fn default() -> Self {
        Self::from([0; N])
    }
}

impl<const N: usize> Dimension for StaticDim<N> {
    const RANK: Option<usize> = Some(N);

    #[inline]
    fn zero(rank: usize) -> Option<Self> {
        if rank == N {
            Some(StaticDim([0; N]))
        } else {
            None
        }
    }

    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension,
    {
        if other.rank() == N {
            other.as_slice().try_into().map(StaticDim).ok()
        } else {
            None
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        self.0.as_ref()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.0.as_mut()
    }

    #[inline]
    fn rank(&self) -> usize {
        N
    }
}

impl<const N: usize> StaticDim<N> {
    /// Creates a new static dimension from an array.
    pub fn from_array(value: [usize; N]) -> Self {
        Self(value)
    }
}

/// `u8` bounded to `[0, MAX_INLINE_RANK]`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
#[repr(u8)]
#[allow(missing_docs)]
enum StaticLen {
    #[default]
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}

impl StaticLen {
    /// Returns the equivalent `StaticLen`, or `None` if it exceeds
    /// [`MAX_INLINE_RANK`].
    fn from_u8(n: u8) -> Option<Self> {
        match n {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            _ => None,
        }
    }
}

/// Multidimensional quantity representation with a size determined at runtime.
#[derive(Clone, Debug)]
enum DynDimInner {
    /// On the stack for up to [`MAX_INLINE_RANK`].
    Stack(StaticLen, [usize; MAX_INLINE_RANK]),
    /// On the heap for higher dimensions.
    Heap(Vec<usize>),
}

impl Deref for DynDimInner {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Stack(n, dims) => &dims[..*n as usize],
            Self::Heap(dims) => dims,
        }
    }
}

impl DerefMut for DynDimInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Stack(n, dims) => &mut dims[..*n as usize],
            Self::Heap(dims) => dims,
        }
    }
}

impl PartialEq for DynDimInner {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for DynDimInner {}

impl std::hash::Hash for DynDimInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl Default for DynDimInner {
    fn default() -> Self {
        Self::Stack(StaticLen::Zero, [0; MAX_INLINE_RANK])
    }
}

impl DynDimInner {
    /// Creates a new dynamic dimension from an array.
    fn from_array<const N: usize>(value: [usize; N]) -> Self {
        if N <= MAX_INLINE_RANK {
            let len = StaticLen::from_u8(N as u8).expect("above `if` should make this infallible");
            let mut dims = [0; MAX_INLINE_RANK];
            dims[..value.len()].copy_from_slice(&value);

            Self::Stack(len, dims)
        } else {
            Self::Heap(value.to_vec())
        }
    }

    /// Creates a new dynamic dimension from a slice.
    fn from_slice<S>(value: S) -> Self
    where
        S: AsRef<[usize]> + Into<Vec<usize>>,
    {
        if value.as_ref().len() <= MAX_INLINE_RANK {
            let len = StaticLen::from_u8(value.as_ref().len() as u8)
                .expect("above `if` should make this infallible");
            let mut dims = [0; MAX_INLINE_RANK];
            dims[..value.as_ref().len()].copy_from_slice(value.as_ref());

            Self::Stack(len, dims)
        } else {
            Self::Heap(value.into())
        }
    }
}

/// Multidimensional quantity with a size determined at runtime.
#[derive(Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct DynDim(DynDimInner);

impl<const N: usize> From<[usize; N]> for DynDim {
    fn from(value: [usize; N]) -> Self {
        Self::from_array(value)
    }
}

impl From<&[usize]> for DynDim {
    fn from(value: &[usize]) -> Self {
        Self::from_slice(value)
    }
}

impl From<Vec<usize>> for DynDim {
    fn from(value: Vec<usize>) -> Self {
        Self::from_slice(value)
    }
}

impl Dimension for DynDim {
    const RANK: Option<usize> = None;

    fn zero(rank: usize) -> Option<Self> {
        if rank <= MAX_INLINE_RANK {
            let len =
                StaticLen::from_u8(rank as u8).expect("above `if` should make this infallible");

            Some(Self(DynDimInner::Stack(len, [0; MAX_INLINE_RANK])))
        } else {
            Some(Self(DynDimInner::Heap(vec![0; rank])))
        }
    }

    fn from_dimension<D>(other: &D) -> Option<Self>
    where
        D: Dimension,
    {
        Some(Self(DynDimInner::from_slice(other.as_slice())))
    }

    fn as_slice(&self) -> &[usize] {
        self.0.deref()
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        self.0.deref_mut()
    }
}

impl DynDim {
    /// Creates a new dynamic dimension from an array.
    pub fn from_array<const N: usize>(value: [usize; N]) -> Self {
        Self(DynDimInner::from_array(value))
    }

    /// Creates a new dynamic dimension from a slice.
    pub fn from_slice<S>(value: S) -> Self
    where
        S: AsRef<[usize]> + Into<Vec<usize>>,
    {
        Self(DynDimInner::from_slice(value))
    }
}

/// Priority order of the dimensions of an array.
///
/// An order is always a permutation of `0..rank`, so no dimension is visited
/// twice or skipped, and indexing a slice of that rank by any entry is
/// infallible.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct DimOrder<D>(D);

impl<D> DimOrder<D>
where
    D: Dimension,
{
    /// Creates a new dimension order.
    ///
    /// Returns `None` if `order` is not a permutation of `0..rank`.
    pub fn new(order: D) -> Option<Self> {
        if (0..order.rank()).all(|dim| order.as_slice().contains(&dim)) {
            Some(Self(order))
        } else {
            None
        }
    }

    /// Returns the order in which the last dimension varies fastest, that is
    /// `0, 1, ..., rank - 1`.
    ///
    /// Returns `None` if `D` cannot represent `rank`.
    pub fn lexicographic(rank: usize) -> Option<Self> {
        let mut order = D::zero(rank)?;
        order
            .as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(dim, slot)| *slot = dim);

        Some(Self(order))
    }

    /// Returns the order in which the first dimension varies fastest, that is
    /// `rank - 1, ..., 1, 0`.
    ///
    /// Returns `None` if `D` cannot represent `rank`.
    pub fn colexicographic(rank: usize) -> Option<Self> {
        let mut order = D::zero(rank)?;
        order
            .as_mut_slice()
            .iter_mut()
            .rev()
            .enumerate()
            .for_each(|(dim, slot)| *slot = dim);

        Some(Self(order))
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns the dimension at the given position in the order.
    ///
    /// Position `0` is the slowest varying dimension. Returns `None` if
    /// `index` is not less than the rank.
    pub fn get(&self, index: usize) -> Option<DimIndex> {
        self.0
            .as_slice()
            .get(index)
            .map(|&dim| DimIndex(dim))
    }

    /// Returns an iterator over the dimensions from slowest to fastest
    /// varying.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = DimIndex> + ExactSizeIterator {
        self.0.as_slice().iter().map(|&dim| DimIndex(dim))
    }

    /// Returns a slice containing all dimension indices.
    pub(crate) fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Returns the order that maps each dimension to its position in `self`.
    pub fn inverse(&self) -> Self {
        let mut inverse = D::zero(self.rank()).expect("D can always represent its own rank");
        let slots = inverse.as_mut_slice();
        for (position, dim) in self.iter().enumerate() {
            slots[dim.0] = position;
        }

        Self(inverse)
    }
}

/// Multidimensional index into an array.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct ArrayIndex<D>(D);

impl<D, const N: usize> From<[usize; N]> for ArrayIndex<D>
where
    D: Dimension + From<[usize; N]>,
{
    fn from(value: [usize; N]) -> Self {
        Self(D::from(value))
    }
}

impl<D> ArrayIndex<D>
where
    D: Dimension,
{
    /// Creates a new multidimensional array index.
    pub fn new(indices: D) -> Self {
        Self(indices)
    }

    /// Returns the equivalent index with a rank determined at runtime.
    pub fn into_dyn(self) -> ArrayIndex<DynDim> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent index of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Option<ArrayIndex<StaticDim<N>>> {
        self.to_dimension()
    }

    /// Returns the `linear`-th index of `shape` in lexicographic order.
    ///
    /// `linear` must be less than the product of `shape`'s extents. Otherwise,
    /// the leading component of the result exceeds its extent.
    pub(crate) fn linear_in_shape(linear: usize, shape: &Shape<D>) -> Self {
        let mut index =
            Self::new(D::zero(shape.rank()).expect("`D` can always represent its own rank"));
        index.set_linear(linear, shape.as_slice());

        index
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns the array index at the specified `DimIndex`.
    pub fn get(&self, index: DimIndex) -> Option<usize> {
        self.0.as_slice().get(index.0).copied()
    }

    /// Returns a slice containing all array indices.
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Returns a mutable slice containing all array indices.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.0.as_mut_slice()
    }

    /// Increments the index according to the `extents` in lexicographic order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    ///
    /// Wraps around when reaching the last index.
    pub(crate) fn increment_lexicographic(&mut self, extents: &[usize]) {
        debug_assert_eq!(self.rank(), extents.len());

        let components = self.0.as_mut_slice();
        for (component, &extent) in components.iter_mut().zip(extents).rev() {
            *component += 1;
            if *component < extent {
                break;
            }
            *component = 0;
        }
    }

    /// Decrements the index according to the `extents` in lexicographic order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    ///
    /// Wraps around when reaching the first index.
    pub(crate) fn decrement_lexicographic(&mut self, extents: &[usize]) {
        debug_assert_eq!(self.rank(), extents.len());

        let components = self.0.as_mut_slice();
        for (component, &extent) in components.iter_mut().zip(extents).rev() {
            if *component > 0 {
                *component -= 1;
                break;
            }
            *component = extent.saturating_sub(1);
        }
    }

    /// Sets the index to the `k`-th one in lexicographic order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    pub(crate) fn set_linear(&mut self, mut linear: usize, extents: &[usize]) {
        debug_assert_eq!(self.rank(), extents.len());
        debug_assert!(
            extents
                .iter()
                .try_fold(1usize, |acc, &e| acc.checked_mul(e))
                .is_none_or(|p| linear == 0 || linear < p)
        );

        let components = self.0.as_mut_slice();
        for (component, &extent) in components.iter_mut().zip(extents).rev() {
            if extent == 0 {
                *component = 0;
            } else {
                *component = linear % extent;
                linear /= extent;
            }
        }
    }
}

impl<D1> ArrayIndex<D1>
where
    D1: Dimension,
{
    /// Returns the equivalent index over `D2`.
    ///
    /// Returns `None` if `D2` cannot represent the rank of `self`.
    pub fn to_dimension<D2>(&self) -> Option<ArrayIndex<D2>>
    where
        D2: Dimension,
    {
        Some(ArrayIndex(D2::from_dimension(&self.0)?))
    }
}

/// Shape of an array.
///
/// The entries represent the extent of the array along each dimension.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Shape<D>(D);

impl<D, const N: usize> From<[usize; N]> for Shape<D>
where
    D: Dimension + From<[usize; N]>,
{
    fn from(value: [usize; N]) -> Self {
        Self(D::from(value))
    }
}

impl<D> Shape<D>
where
    D: Dimension,
{
    /// Creates a new array shape.
    pub fn new(shape: D) -> Self {
        Self(shape)
    }

    /// Returns the equivalent shape with a rank determined at runtime.
    pub fn into_dyn(self) -> Shape<DynDim> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent shape of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Option<Shape<StaticDim<N>>> {
        self.to_dimension()
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns the array extent at the specified `DimIndex`.
    pub fn get(&self, index: DimIndex) -> Option<usize> {
        self.0.as_slice().get(index.0).copied()
    }

    /// Returns a slice containing all array extents.
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Returns a mutable slice containing all array extents.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.0.as_mut_slice()
    }

    /// Returns the product of array extents, or `None` if overflow occurred.
    pub fn product_checked(&self) -> Option<usize> {
        self.0
            .as_slice()
            .iter()
            .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
    }

    /// Computes the row-major, contiguous strides from the array shape.
    pub fn row_major_strides(&self) -> Option<Strides<D>> {
        let mut strides = D::zero(self.0.rank()).expect("D can always represent its own rank");
        if let Some(last) = strides.as_mut_slice().last_mut() {
            let mut product = 1;
            *last = product;
            for (stride, &extent) in strides
                .as_mut_slice()
                .iter_mut()
                .rev()
                .skip(1)
                .zip(self.0.as_slice().iter().rev())
            {
                product = product.checked_mul(extent)?;
                *stride = product;
            }
        }

        Some(Strides(strides))
    }

    /// Computes the column-major, contiguous strides from the array shape.
    pub fn column_major_strides(&self) -> Option<Strides<D>> {
        let mut strides = D::zero(self.0.rank()).expect("`D` can always represent its own rank");
        if let Some(first) = strides.as_mut_slice().first_mut() {
            let mut product = 1;
            *first = product;
            for (stride, &extent) in strides
                .as_mut_slice()
                .iter_mut()
                .skip(1)
                .zip(self.0.as_slice())
            {
                product = product.checked_mul(extent)?;
                *stride = product;
            }
        }

        Some(Strides(strides))
    }

    /// Returns an iterator over the multidimensional indices in lexicographic
    /// order.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    pub fn indices_lexicographic(&self) -> Option<Indices<D>> {
        Indices::new(self.clone())
    }

    /// Returns a parallel iterator over the multidimensional indices in
    /// lexicographic order.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    #[cfg(feature = "rayon")]
    pub fn par_indices_lexicographic(&self) -> Option<ParIndices<D>> {
        ParIndices::new(self.clone())
    }
}

impl<D1> Shape<D1>
where
    D1: Dimension,
{
    /// Returns the equivalent shape over `D2`.
    ///
    /// Returns `None` if `D2` cannot represent the rank of `self`.
    pub fn to_dimension<D2>(&self) -> Option<Shape<D2>>
    where
        D2: Dimension,
    {
        Some(Shape(D2::from_dimension(&self.0)?))
    }
}

/// Strides of the elements of an array.
///
/// The entries represent how far apart elements along a dimension are in the
/// contiguous buffer.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Strides<D>(D);

impl<D> Strides<D>
where
    D: Dimension,
{
    /// Creates a new collection of array strides.
    pub fn new(strides: D) -> Self {
        Self(strides)
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns the element stride at the specified `DimIndex`.
    pub fn get(&self, index: DimIndex) -> Option<usize> {
        self.0.as_slice().get(index.0).copied()
    }

    /// Returns a slice containing all array element strides.
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }
}

/// Layout of an array.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Layout<D> {
    /// Array shape.
    ///
    /// A `&mut Shape<D>` must *never* escape to anywhere. Otherwise, all
    /// established invariants may be broken.
    shape: Shape<D>,
    /// Array strides.
    strides: Strides<D>,
    /// Offset from the start of the buffer.
    offset: usize,
    /// Largest buffer offset this layout can address.
    max_offset: usize,
    /// Element count of the layout.
    len: usize,
}

impl<D> Layout<D>
where
    D: Dimension,
{
    /// Creates a new layout.
    ///
    /// Returns `None` if `shape` and `strides` have different rank, or if
    /// [`Shape::product_checked`] returns `Some(0)` or `None`, or if computing
    /// the maximum offset that can be addressed overflows.
    pub fn new(shape: Shape<D>, strides: Strides<D>, offset: usize) -> Option<Self> {
        if shape.rank() != strides.rank() {
            return None;
        }

        let len = shape
            .product_checked()
            .filter(|&size| size != 0)?;
        let max_offset = Self::max_offset_of(&shape, &strides, offset)?;

        Some(Self {
            shape,
            strides,
            offset,
            max_offset,
            len,
        })
    }

    /// Creates a row-major, contiguous layout with the given shape and offset.
    ///
    /// Returns `None` if [`Shape::product_checked`] returns `Some(0)` or
    /// `None`, or if no contiguous strides can be computed from `shape`, or if
    /// computing the maximum offset that can be addressed overflows.
    pub fn row_major(shape: Shape<D>, offset: usize) -> Option<Self> {
        let len = shape
            .product_checked()
            .filter(|&size| size != 0)?;
        let strides = shape.row_major_strides()?;
        let max_offset = Self::max_offset_of(&shape, &strides, offset)?;

        Some(Self {
            shape,
            strides,
            offset,
            max_offset,
            len,
        })
    }

    /// Creates a column-major, contiguous layout with the given shape and
    /// offset.
    ///
    /// Returns `None` if [`Shape::product_checked`] returns `Some(0)` or
    /// `None`, or if no contiguous strides can be computed from `shape`, or if
    /// computing the maximum offset that can be addressed overflows.
    ///
    /// The library stores arrays row-major throughout. See
    /// [`Shape::column_major_strides`].
    pub fn column_major(shape: Shape<D>, offset: usize) -> Option<Self> {
        let len = shape
            .product_checked()
            .filter(|&size| size != 0)?;
        let strides = shape.column_major_strides()?;
        let max_offset = Self::max_offset_of(&shape, &strides, offset)?;

        Some(Self {
            shape,
            strides,
            offset,
            max_offset,
            len,
        })
    }

    /// Returns the layout with its dimensions reordered according to `order`.
    ///
    /// Dimension `i` of the result is dimension `order[i]` of `self`, for both
    /// extents and strides. Permuting by [`Layout::memory_order`] yields the
    /// layout whose lexicographic traversal is the most sequential one.
    ///
    /// Returns `None` if `order` has a different rank than `self`.
    pub fn permuted(&self, order: &DimOrder<D>) -> Option<Self> {
        if order.rank() != self.rank() {
            return None;
        }

        let zero = D::zero(self.rank()).expect("`D` can always represent its own rank");
        let mut shape = Shape(zero.clone());
        let mut strides = Strides(zero);
        let (extents, old_strides) = (self.shape.as_slice(), self.strides.as_slice());
        for (position, dim) in order.iter().enumerate() {
            shape.as_mut_slice()[position] = extents[dim.0];
            strides.0.as_mut_slice()[position] = old_strides[dim.0];
        }

        Some(Self {
            shape,
            strides,
            // these carry over as is (pinky promise) because reordering the
            // exclusively finite, non-negative terms of a finite sum/product
            // can't make it overflow if the original didn't overflow (which
            // all other constructors guarantee) and does not change it
            offset: self.offset,
            max_offset: self.max_offset,
            len: self.len,
        })
    }

    /// Returns the equivalent layout with a rank determined at runtime.
    pub fn into_dyn(self) -> Layout<DynDim> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent layout of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Option<Layout<StaticDim<N>>> {
        self.to_dimension()
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns a reference to the shape of the layout.
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Returns a reference to the strides of the layout.
    pub fn strides(&self) -> &Strides<D> {
        &self.strides
    }

    /// Returns the offset from the start of the buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of elements contained in the layout.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the layout contains no elements.
    ///
    /// This always returns `false`, since no constructor produces an empty
    /// layout. It exists as the counterpart to [`Layout::len`].
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Largest buffer offset this layout can address.
    pub fn max_offset(&self) -> usize {
        self.max_offset
    }

    /// Returns `true` if the layout covers `offset..offset + len` with no gaps.
    ///
    /// Dimensions with an extent of 1 are ignored, since their stride is never
    /// used to address an element.
    pub fn is_packed(&self) -> bool {
        let extents = self.shape.as_slice();
        let strides = self.strides.as_slice();

        let mut expected = 1_usize;
        for dim in self.memory_order().iter().rev() {
            let (extent, stride) = (extents[dim.0], strides[dim.0]);
            if extent == 1 {
                continue;
            }
            if stride != expected {
                return false;
            }
            expected *= extent;
        }

        true
    }

    /// Returns `true` if the strides are exactly [`Shape::row_major_strides`].
    pub fn is_row_major(&self) -> bool {
        self.shape
            .row_major_strides()
            .is_some_and(|row_major_strides| row_major_strides == self.strides)
    }

    /// Returns `true` if no two indices address the same buffer offset.
    pub fn is_non_overlapping(&self) -> bool {
        let extents = self.shape.as_slice();
        let strides = self.strides.as_slice();

        let mut expected = 1_usize;
        for dim in self.memory_order().iter().rev() {
            let (extent, stride) = (extents[dim.0], strides[dim.0]);
            if extent == 1 {
                continue;
            }
            if stride < expected {
                return false;
            }
            expected = stride.saturating_mul(extent);
        }

        true
    }

    /// Returns the last dimension with the smallest stride.
    ///
    /// Useful for choosing a processing dimension when none is specified, since
    /// a lane along this dimension is the cheapest to traverse.
    ///
    /// Dimensions with an extent of 1 are ignored, since their stride never
    /// addresses an element. Ties are broken toward the later dimension.
    ///
    /// Returns `None` if every extent is 1, including for rank 0.
    pub fn fastest_dimension(&self) -> Option<DimIndex> {
        self.shape
            .as_slice()
            .iter()
            .zip(self.strides.as_slice())
            .enumerate()
            .filter(|&(_, (&extent, _))| extent > 1)
            // breaks ties by picking the last maximum (minimum with `Reverse`)
            .max_by_key(|&(_, (_, &stride))| std::cmp::Reverse(stride))
            .map(|(index, _)| DimIndex(index))
    }

    /// Returns the first dimension along which lanes are contiguous in the
    /// buffer.
    ///
    /// This is the dimension with stride 1, ignoring extents of 1, whose lanes
    /// are therefore the cheapest possible to traverse.
    ///
    /// Returns `None` if no dimension has stride 1, which includes rank 0.
    pub fn contiguous_dimension(&self) -> Option<DimIndex> {
        self.shape
            .as_slice()
            .iter()
            .zip(self.strides.as_slice())
            .enumerate()
            .find(|&(_, (&extent, &stride))| extent > 1 && stride == 1)
            .map(|(index, _)| DimIndex(index))
    }

    /// Returns the lexicographic dimension order for `D` with the same rank as
    /// `self`.
    pub fn lexicographic_order(&self) -> DimOrder<D> {
        DimOrder::lexicographic(self.rank()).expect("D can always represent its own rank")
    }

    /// Returns the dimension indices ordered from largest to smallest stride.
    ///
    /// Walking dimensions in this order visits the buffer as sequentially as
    /// the layout permits. Ties are broken by ascending dimension index, so a
    /// row-major layout yields [`DimOrder::lexicographic`], while a
    /// column-major layout yields [`DimOrder::colexicographic`].
    pub fn memory_order(&self) -> DimOrder<D> {
        let rank = self.rank();
        let strides = self.strides.as_slice();
        let mut order = D::zero(rank).expect("D can always represent its own rank");
        let slots = order.as_mut_slice();
        slots
            .iter_mut()
            .enumerate()
            .for_each(|(index, slot)| *slot = index);
        for i in 1..rank {
            let mut j = i;
            while j > 0 && strides[slots[j - 1]] < strides[slots[j]] {
                slots.swap(j - 1, j);
                j -= 1;
            }
        }

        DimOrder(order)
    }

    /// Returns an iterator over the lanes in memory order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn lanes_memory_order(&self, dim: DimIndex) -> Option<LaneGeometries<D>> {
        LaneGeometries::new(self.clone(), dim, self.memory_order())
    }

    /// Returns an iterator over the lanes in lexicographic order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn lanes_lexicographic(&self, dim: DimIndex) -> Option<LaneGeometries<D>> {
        LaneGeometries::new(self.clone(), dim, self.lexicographic_order())
    }

    /// Returns an iterator over the lanes in the provided order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`.
    pub fn lanes_with_order(&self, dim: DimIndex, order: DimOrder<D>) -> Option<LaneGeometries<D>> {
        LaneGeometries::new(self.clone(), dim, order)
    }

    /// Returns a parallel iterator over the lanes in memory order.
    ///
    /// Returns `None` if `dim` is out of range.
    #[cfg(feature = "rayon")]
    pub fn par_lanes_memory_order(&self, dim: DimIndex) -> Option<ParLaneGeometries<D>> {
        ParLaneGeometries::new(self.clone(), dim, self.memory_order())
    }

    /// Returns a parallel iterator over the lanes in lexicographic order.
    ///
    /// Returns `None` if `dim` is out of range.
    #[cfg(feature = "rayon")]
    pub fn par_lanes_lexicographic(&self, dim: DimIndex) -> Option<ParLaneGeometries<D>> {
        ParLaneGeometries::new(self.clone(), dim, self.lexicographic_order())
    }

    /// Returns a parallel iterator over the lanes in the provided order.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// Returns `None` if `dim` is out of range, or if `order` has a different
    /// rank than `self`.
    #[cfg(feature = "rayon")]
    pub fn par_lanes_with_order(
        &self,
        dim: DimIndex,
        order: DimOrder<D>,
    ) -> Option<ParLaneGeometries<D>> {
        ParLaneGeometries::new(self.clone(), dim, order)
    }

    /// Returns the number of lanes along `dim`.
    ///
    /// A lane is the one-dimensional sequence of elements obtained by varying
    /// the index along `dim` while holding all other indices fixed. The count
    /// is therefore [`Layout::len`] divided by the extent along `dim`.
    ///
    /// Returns `None` if `dim` is out of range, which includes every `dim` at
    /// rank 0.
    pub fn lane_count(&self, dim: DimIndex) -> Option<usize> {
        self.shape
            .get(dim)
            .map(|extent| self.len / extent)
    }

    /// Returns `true` if the lanes along `dim` in memory order are consecutive,
    /// contiguous chunks of the underlying storage.
    ///
    /// When this holds, lanes can straightforwardly be handed out as disjoint,
    /// potentially mutable slices without any offset arithmetic.
    ///
    /// Always returns `false` if `dim` is out of range.
    pub fn lanes_are_chunks(&self, dim: DimIndex) -> bool {
        let Some(extent) = self.shape.get(dim) else {
            return false;
        };
        let Some(stride) = self.strides.get(dim) else {
            return false;
        };

        (extent == 1 || stride == 1) && self.is_packed()
    }

    /// Returns the geometry of the `lane`-th lane along `dim`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// `dim` must be in range, `order` must have the same rank as the layout,
    /// and `lane` must be less than [`Layout::lane_count`]. Otherwise, the
    /// result is meaningless.
    pub(crate) fn lane_unvalidated(
        &self,
        dim: DimIndex,
        lane: usize,
        order: &DimOrder<D>,
    ) -> LaneGeometry {
        // upholds the variant because every offset in a lane geometry is
        // bounded from above by `Layout::max_offset`, which doesn't overflow
        // by this type's invariants.
        LaneGeometry {
            offset: self.lane_offset_unvalidated(dim, lane, order),
            stride: self.strides.as_slice()[dim.0],
            count: self.shape.as_slice()[dim.0],
        }
    }

    /// Returns the buffer offset of the first element of the `lane`-th lane
    /// along `dim`.
    ///
    /// Lanes are numbered according to `order` over the other dimensions.
    ///
    /// `dim` must be in range and `lane` must be less than
    /// [`lane_count`](Self::lane_count). Otherwise, the result is meaningless.
    ///
    /// No arithmetic can overflow under those preconditions: every offset this
    /// produces is at most [`Layout::max_offset`], which was validated at
    /// construction.
    pub(crate) fn lane_offset_unvalidated(
        &self,
        dim: DimIndex,
        mut lane: usize,
        order: &DimOrder<D>,
    ) -> usize {
        debug_assert!(dim.0 < self.rank());
        debug_assert_eq!(order.rank(), self.rank());
        debug_assert!(
            self.lane_count(dim)
                .is_some_and(|count| lane < count)
        );

        let extents = self.shape.as_slice();
        let strides = self.strides.as_slice();

        order
            .as_slice()
            .iter()
            .rev()
            .copied()
            .filter(|&d| d != dim.0)
            .fold(self.offset, |acc, d| {
                let acc = acc + (lane % extents[d]) * strides[d];
                lane /= extents[d];

                acc
            })
    }

    /// Returns the largest buffer offset addressable by `shape` and `strides`
    /// starting at `offset`, or `None` if the computation overflows or any
    /// extent is zero.
    fn max_offset_of(shape: &Shape<D>, strides: &Strides<D>, offset: usize) -> Option<usize> {
        shape
            .as_slice()
            .iter()
            .zip(strides.as_slice())
            .try_fold(offset, |acc, (&extent, &stride)| {
                acc.checked_add(extent.checked_sub(1)?.checked_mul(stride)?)
            })
    }
}

impl<D1> Layout<D1>
where
    D1: Dimension,
{
    /// Returns the equivalent layout over `D2`.
    ///
    /// Returns `None` if `D2` cannot represent the rank of `self`.
    pub fn to_dimension<D2>(&self) -> Option<Layout<D2>>
    where
        D2: Dimension,
    {
        Some(Layout {
            shape: Shape(D2::from_dimension(&self.shape.0)?),
            strides: Strides(D2::from_dimension(&self.strides.0)?),
            offset: self.offset,
            max_offset: self.max_offset,
            len: self.len,
        })
    }

    /// Returns the linear buffer offset of `index`.
    ///
    /// Returns `None` if `index` has a different rank than the layout, or if
    /// any component is out of bounds.
    pub fn linear<D2>(&self, index: &ArrayIndex<D2>) -> Option<usize>
    where
        D2: Dimension,
    {
        if index.rank() != self.shape.rank() {
            return None;
        }

        index
            .as_slice()
            .iter()
            .zip(self.shape.as_slice())
            .zip(self.strides.as_slice())
            .try_fold(self.offset, |acc, ((&index, &extent), &stride)| {
                if index >= extent {
                    return None;
                }

                acc.checked_add(index.checked_mul(stride)?)
            })
    }

    /// Returns the linear buffer offset of `index`, eliding any checks.
    ///
    /// In particular, it must hold that `index` and layout have the same rank,
    /// and none of the components of `index` is out of bounds. If any of these
    /// conditions do not hold, the result is meaningless.
    pub fn linear_unvalidated<D2>(&self, index: &ArrayIndex<D2>) -> usize
    where
        D2: Dimension,
    {
        debug_assert_eq!(self.rank(), index.rank());

        index
            .as_slice()
            .iter()
            .zip(self.strides.as_slice())
            .fold(self.offset, |acc, (&index, &stride)| {
                acc.wrapping_add(index.wrapping_mul(stride))
            })
    }

    /// Returns the lane along `dim` that passes through `index`.
    ///
    /// The component of `index` at `dim` is ignored, so an index anywhere on
    /// the lane selects it. Unlike the numbering used by the `lanes_*`
    /// iterators, this identifies a lane absolutely and needs no order.
    ///
    /// Returns `None` if `dim` is out of range, if `index` has a different
    /// rank than the layout, or if any other component is out of bounds.
    pub fn lane_at<D2>(&self, dim: DimIndex, index: &ArrayIndex<D2>) -> Option<LaneGeometry>
    where
        D2: Dimension,
    {
        if index.rank() != self.rank() || dim.0 >= self.rank() {
            return None;
        }

        let (extents, strides) = (self.shape.as_slice(), self.strides.as_slice());
        let offset = index
            .as_slice()
            .iter()
            .zip(extents)
            .zip(strides)
            .enumerate()
            .filter(|&(component, _)| component != dim.0)
            .try_fold(self.offset, |acc, (_, ((&index, &extent), &stride))| {
                (index < extent).then(|| acc + index * stride)
            })?;

        // upholds the variant because every offset in a lane geometry is
        // bounded from above by `Layout::max_offset`, which doesn't overflow
        // by this type's invariants.
        Some(LaneGeometry {
            offset,
            stride: strides[dim.0],
            count: extents[dim.0],
        })
    }
}

/// Geometry of a single lane through an array buffer.
///
/// Describes the `count` buffer offsets `offset`, `offset + stride`, ...,
/// `offset + (count - 1) * stride`. Carries no dimension information: a lane
/// is a flat walk, and the layout it came from determines what it traverses.
///
/// # Invariants
///
/// `offset + (count - 1) * stride` does not overflow. Since strides are
/// unsigned, this makes every addressable offset non-overflowing, and the
/// offsets monotonically increasing in the index.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LaneGeometry {
    /// Offset of the first element.
    offset: usize,
    /// Number of elements between consecutive elements of the lane.
    stride: usize,
    /// Number of elements in the lane.
    count: usize,
}

impl LaneGeometry {
    /// Creates a new lane geometry.
    ///
    /// Returns `None` if `offset + (count - 1) * stride` overflows.
    #[inline]
    pub fn new(offset: usize, stride: usize, count: usize) -> Option<Self> {
        if offset
            .checked_add(count.checked_sub(1)?.checked_mul(stride)?)
            .is_some()
        {
            Some(LaneGeometry {
                offset,
                stride,
                count,
            })
        } else {
            None
        }
    }

    /// Returns the offset of the first element in the lane.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the stride between lane elements.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the number of lane elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the lane geometry addresses no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if traversing the geometry only yields unique offsets.
    #[inline]
    pub fn is_injective(&self) -> bool {
        self.stride > 0 || self.count <= 1
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == 1 || self.count <= 1
    }

    /// Returns the range of contiguous buffer offsets addressed by the lane
    /// geometry.
    #[inline]
    pub fn contiguous_range(&self) -> Option<std::ops::Range<usize>> {
        if self.is_contiguous() {
            Some(self.offset..self.offset.checked_add(self.count)?)
        } else {
            None
        }
    }

    /// Returns the largest buffer offset the lane addresses.
    ///
    /// Returns `None` if the lane is empty, and therefore addresses nothing, or
    /// if the computation overflows.
    #[inline]
    pub fn max_offset(&self) -> Option<usize> {
        let last = self.count.checked_sub(1)?;

        self.offset
            .checked_add(last.checked_mul(self.stride)?)
    }

    /// Returns `true` if every offset the lane addresses is less than `len`.
    ///
    /// An empty lane always fits. A lane whose offsets overflow never does.
    #[inline]
    pub fn fits_within(&self, len: usize) -> bool {
        if self.count == 0 {
            true
        } else {
            self.max_offset().is_some_and(|max| max < len)
        }
    }

    /// Returns the buffer offset of the `index`-th element.
    ///
    /// Returns `None` if `index` is out of bounds or if the computation
    /// overflows.
    #[inline]
    pub fn offset_of(&self, index: usize) -> Option<usize> {
        if index < self.count {
            self.offset
                .checked_add(index.checked_mul(self.stride)?)
        } else {
            None
        }
    }

    /// Returns the buffer offset of the `index`-th element.
    ///
    /// `index` must be addressable by the lane.
    #[inline]
    pub fn offset_of_unvalidated(&self, index: usize) -> usize {
        debug_assert!(index < self.count);

        self.offset + index * self.stride
    }

    /// Returns an iterator over the buffer offsets addressed by the lane
    /// geometry.
    ///
    /// If the lane geometry addresses contiguous slices of the buffer, prefer
    /// using [`LaneGeometry::contiguous_range`] as an iterator.
    #[inline]
    pub fn offsets(self) -> LaneOffsets {
        LaneOffsets::new(self)
    }

    /// Returns a parallel iterator over the buffer offsets addressed by the
    /// lane geometry.
    ///
    /// If the lane geometry addresses contiguous slices of the buffer, prefer
    /// using [`LaneGeometry::contiguous_range`] as an iterator.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_offsets(self) -> LaneOffsets {
        LaneOffsets::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    /// `(extents, contiguous strides, element count)`.
    const CONTIGUOUS: [(&[usize], &[usize], usize); 9] = [
        (&[], &[], 1),
        (&[1], &[1], 1),
        (&[5], &[1], 5),
        (&[2, 3], &[3, 1], 6),
        (&[2, 3, 4], &[12, 4, 1], 24),
        (&[2, 3, 4, 5], &[60, 20, 5, 1], 120),
        (&[1, 7, 1], &[7, 1, 1], 7),
        (&[7, 1, 1], &[1, 1, 1], 7),
        (&[1, 1, 7], &[7, 7, 1], 7),
    ];

    #[test]
    fn static_dynamic_bridge() {
        let dim0 = DynDim::from_array([]);
        let dim1 = DynDim::from_array([1]);
        let dim2 = DynDim::from_array([1, 2]);
        let dim3 = DynDim::from_array([1, 2, 3]);
        let dim4 = DynDim::from_array([1, 2, 3, 4]);
        let dim5 = DynDim::from_array([1, 2, 3, 4, 5]);

        assert!(StaticDim::<0>::from_dimension(&dim0).is_some());
        assert!(StaticDim::<1>::from_dimension(&dim1).is_some());
        assert!(StaticDim::<2>::from_dimension(&dim2).is_some());
        assert!(StaticDim::<3>::from_dimension(&dim3).is_some());
        assert!(StaticDim::<4>::from_dimension(&dim4).is_some());
        assert!(StaticDim::<5>::from_dimension(&dim5).is_some());
    }

    #[test]
    fn zero_respects_rank() {
        assert!(StaticDim::<3>::zero(0).is_none());
        assert!(StaticDim::<3>::zero(1).is_none());
        assert!(StaticDim::<3>::zero(2).is_none());
        assert!(StaticDim::<3>::zero(3).is_some());
        assert!(StaticDim::<3>::zero(4).is_none());
        assert!(StaticDim::<3>::zero(5).is_none());
        assert_eq!(StaticDim::<3>::zero(3).unwrap().as_slice(), &[0; 3]);
        assert_eq!(DynDim::zero(9).unwrap().as_slice(), &[0; 9]);
    }

    #[test]
    fn dyn_dim_stack_heap() {
        for rank in 0..=MAX_INLINE_RANK {
            assert!(matches!(
                DynDim::zero(rank).unwrap().0,
                DynDimInner::Stack(..)
            ));
        }
        for rank in (MAX_INLINE_RANK + 1)..20 {
            assert!(matches!(
                DynDim::zero(rank).unwrap().0,
                DynDimInner::Heap(..)
            ));
        }
    }

    #[test]
    fn conversion() {
        let stat = StaticDim::from([1, 2, 3]);
        let converted = DynDim::from_dimension(&stat).expect("must never fail");
        let recovered = StaticDim::from_dimension(&converted).expect("round trip must never fail");

        assert_eq!(stat.as_slice(), converted.as_slice());
        assert_eq!(stat, recovered);

        let dynamic = DynDim::from_array([1, 2, 3, 4, 5]);
        let converted = StaticDim::<5>::from_dimension(&dynamic).expect("must never fail");
        let recovered = DynDim::from_dimension(&converted).expect("must never fail");

        assert_eq!(dynamic.as_slice(), converted.as_slice());
        assert_eq!(dynamic, recovered);
    }

    #[test]
    fn dyn_dim_dirty() {
        let clean = DynDim(DynDimInner::Stack(StaticLen::One, [5, 0, 0]));
        let dirty = DynDim(DynDimInner::Stack(StaticLen::One, [5, 5, 5]));

        assert_eq!(clean, dirty);

        let hash = |d: &DynDim| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            d.hash(&mut h);

            h.finish()
        };

        assert_eq!(hash(&clean), hash(&dirty));
    }

    #[test]
    fn dim_order_validation() {
        for entries in [
            &[0, 0][..],
            &[1, 1][..],
            &[0, 2][..],
            &[1, 2][..],
            &[3, 1, 2][..],
            &[0, 1, 3][..],
        ] {
            assert!(DimOrder::new(DynDim::from_slice(entries)).is_none());
        }
        for entries in [
            &[][..],
            &[0][..],
            &[0, 1][..],
            &[1, 0][..],
            &[0, 1, 2][..],
            &[0, 2, 1][..],
            &[1, 0, 2][..],
            &[1, 2, 0][..],
            &[2, 0, 1][..],
            &[2, 1, 0][..],
        ] {
            assert!(DimOrder::new(DynDim::from_slice(entries)).is_some());
        }

        assert_eq!(
            DimOrder::<DynDim>::lexicographic(3)
                .unwrap()
                .as_slice(),
            &[0, 1, 2]
        );
        assert_eq!(
            DimOrder::<DynDim>::colexicographic(3)
                .unwrap()
                .as_slice(),
            &[2, 1, 0]
        );
        assert_eq!(
            DimOrder::<DynDim>::lexicographic(0)
                .unwrap()
                .as_slice(),
            &[]
        );
        assert_eq!(
            DimOrder::<DynDim>::colexicographic(0)
                .unwrap()
                .as_slice(),
            &[]
        );

        assert!(DimOrder::<StaticDim<3>>::lexicographic(2).is_none());
        assert!(DimOrder::<StaticDim<3>>::lexicographic(3).is_some());
        assert!(DimOrder::<StaticDim<3>>::lexicographic(4).is_none());
    }

    #[test]
    fn dim_order_inverse() {
        let order = DimOrder::new(DynDim::from_array([1, 2, 0])).unwrap();
        let inverse = order.inverse();

        assert_eq!(order.inverse().as_slice(), &[2, 0, 1]);
        assert_eq!(order.inverse().inverse().as_slice(), order.as_slice());
        for canonical in [
            DimOrder::<DynDim>::lexicographic(4).unwrap(),
            DimOrder::<DynDim>::colexicographic(4).unwrap(),
        ] {
            assert_eq!(canonical.inverse().as_slice(), canonical.as_slice());
        }
        for (position, dim) in order.iter().enumerate() {
            assert_eq!(inverse.get(dim.0).unwrap().0, position);
        }
        assert_eq!(order.get(0), Some(DimIndex(1)));
        assert_eq!(order.get(3), None);
        assert_eq!(
            order
                .iter()
                .rev()
                .map(|d| d.0)
                .collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
        assert_eq!(inverse.inverse().as_slice(), order.as_slice());
    }

    #[test]
    fn index_lexicographic_round_trip() {
        let extents = &[2, 3][..];
        let expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]];

        let mut index = ArrayIndex::new(DynDim::from_array([0, 0]));

        for step in expected {
            assert_eq!(index.as_slice(), &step);
            index.increment_lexicographic(extents);
        }
        assert_eq!(index.as_slice(), &[0, 0]);
        for step in expected.iter().rev() {
            index.decrement_lexicographic(extents);
            assert_eq!(index.as_slice(), step);
        }
        assert_eq!(index.as_slice(), &[0, 0]);
    }

    #[test]
    fn index_set_linear() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]];

        for (linear, step) in expected.iter().enumerate() {
            let index = ArrayIndex::linear_in_shape(linear, &shape);
            let mut curr = ArrayIndex::new(DynDim::from_array([0, 0]));
            for _ in 0..linear {
                curr.increment_lexicographic(shape.as_slice());
            }

            assert_eq!(index.as_slice(), step);
            assert_eq!(curr.as_slice(), step);
        }
    }

    #[test]
    fn index_degenerate_dimensions() {
        let extents = &[3, 1, 2][..];
        let expected = [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [2, 0, 0],
            [2, 0, 1],
        ];

        let mut index = ArrayIndex::new(DynDim::from_array([0, 0, 0]));

        for step in expected {
            assert_eq!(index.as_slice(), &step);
            index.increment_lexicographic(extents);
        }
        assert_eq!(index.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn shape_size() {
        for (extents, _, len) in CONTIGUOUS {
            let shape = Shape::new(DynDim::from_slice(extents));

            assert_eq!(shape.product_checked(), Some(len));
        }
    }

    #[test]
    fn rank_zero_scalar() {
        let layout =
            Layout::row_major(Shape::new(DynDim::from_slice(&[][..])), 9).expect("hand verified");

        assert_eq!(layout.len(), 1);
        assert_eq!(layout.shape().as_slice(), &[] as &[usize]);
        assert_eq!(layout.max_offset(), 9);
    }

    #[test]
    fn extent_zero() {
        for extents in [&[0][..], &[0, 5][..], &[5, 0][..], &[4, 0, 2][..]] {
            let shape = Shape::new(DynDim::from_slice(extents));

            assert_eq!(shape.product_checked(), Some(0));
            assert!(Layout::row_major(shape, 0).is_none());
        }
    }

    #[test]
    fn extent_overflow() {
        let half_max = usize::MAX / 2 + 1;
        let shape = Shape::new(DynDim::from_slice(&[half_max, 4][..]));

        assert!(shape.product_checked().is_none());
        assert!(shape.row_major_strides().is_some());
        assert!(Layout::row_major(shape, 0).is_none());

        let shape = Shape::new(DynDim::from_slice(&[2, half_max, 4][..]));

        assert!(shape.product_checked().is_none());
        assert!(shape.row_major_strides().is_none());
        assert!(Layout::row_major(shape, 0).is_none());
    }

    #[test]
    fn offset_overflow() {
        let shape = Shape::new(DynDim::from_slice(&[4][..]));
        assert!(Layout::row_major(shape, usize::MAX - 3).is_some());

        let shape = Shape::new(DynDim::from_slice(&[4][..]));
        assert!(Layout::row_major(shape, usize::MAX - 2).is_none());
    }

    #[test]
    fn index_zero_extent() {
        let shape = Shape::new(DynDim::from_array([2, 0, 3]));

        assert_eq!(
            ArrayIndex::linear_in_shape(0, &shape).as_slice(),
            &[0, 0, 0]
        );
        assert_eq!(shape.indices_lexicographic().unwrap().count(), 0);
    }

    #[test]
    fn indices_rank_zero() {
        let indices = Shape::new(DynDim::from_array([]))
            .indices_lexicographic()
            .unwrap()
            .collect::<Vec<ArrayIndex<DynDim>>>();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].as_slice(), &[] as &[usize]);
    }

    #[test]
    fn indices_interleaved() {
        let mut iter = Shape::new(DynDim::from_array([2, 3]))
            .indices_lexicographic()
            .unwrap();
        let mut seen = Vec::new();

        for take_front in [true, false, true, false, true, false] {
            let index = if take_front {
                iter.next()
            } else {
                iter.next_back()
            };
            seen.push(index.unwrap().as_slice().to_vec());
        }

        assert_eq!(
            seen,
            vec![
                vec![0, 0],
                vec![1, 2],
                vec![0, 1],
                vec![1, 1],
                vec![0, 2],
                vec![1, 0],
            ]
        );
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn contiguous_strides() {
        for (extents, expected, _) in CONTIGUOUS {
            let shape = Shape::new(DynDim::from_slice(extents));
            let strides = shape.row_major_strides().unwrap();

            assert_eq!(strides.as_slice(), expected);
        }
    }

    #[test]
    fn strides_duality() {
        for (extents, _, _) in CONTIGUOUS {
            let reversed = extents
                .iter()
                .rev()
                .copied()
                .collect::<Vec<usize>>();

            let row = Shape::new(DynDim::from_slice(extents))
                .row_major_strides()
                .unwrap();
            let column = Shape::new(DynDim::from_slice(reversed))
                .column_major_strides()
                .unwrap();
            let column = column
                .as_slice()
                .iter()
                .rev()
                .copied()
                .collect::<Vec<usize>>();

            assert_eq!(row.as_slice(), &column);
        }
    }

    #[test]
    fn column_major_strides_overflow() {
        let half_max = usize::MAX / 2 + 1;

        let shape = Shape::new(DynDim::from_array([2, 2, half_max]));

        assert!(shape.product_checked().is_none());
        assert!(shape.column_major_strides().is_some());
        assert!(Layout::column_major(shape, 0).is_none());

        let shape = Shape::new(DynDim::from_array([2, half_max, 2]));

        assert!(shape.column_major_strides().is_none());
    }

    #[test]
    fn contiguous_layout() {
        for (extents, _, len) in CONTIGUOUS {
            let shape = Shape::new(DynDim::from_slice(extents));

            for offset in [0, 7, 1000] {
                let layout = Layout::row_major(shape.clone(), offset).expect("hand verified");
                let last = ArrayIndex::new(DynDim::from_slice(
                    extents
                        .iter()
                        .map(|e| e - 1)
                        .collect::<Vec<usize>>(),
                ));

                assert_eq!(layout.max_offset(), offset + len - 1);
                assert_eq!(layout.linear(&last), Some(layout.max_offset()));
                assert_eq!(layout.linear_unvalidated(&last), layout.max_offset());
            }
        }
    }

    #[test]
    fn layout_linear() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(layout.strides().as_slice(), &[12, 4, 1]);
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 0, 0]))),
            Some(0)
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 0, 1]))),
            Some(1)
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 1, 0]))),
            Some(4)
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([1, 0, 0]))),
            Some(12)
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([1, 2, 3]))),
            Some(23)
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([2, 0, 0]))),
            None
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 3, 0]))),
            None
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 0, 4]))),
            None
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 0]))),
            None
        );
        assert_eq!(
            layout.linear(&ArrayIndex::new(DynDim::from_array([0, 0, 0, 0]))),
            None
        );
    }

    #[test]
    fn offset_overflow_multidimensional() {
        let shape = Shape::new(DynDim::from_array([2, 2]));

        assert!(Layout::row_major(shape.clone(), usize::MAX - 3).is_some());
        assert!(Layout::row_major(shape, usize::MAX - 2).is_none());
    }

    #[test]
    fn layout_validation() {
        let shapes = [
            Shape::new(DynDim::from_array([2, 3])),
            Shape::new(DynDim::from_array([2, 0])),
            Shape::new(DynDim::from_array([2, 2])),
        ];
        let strides = [
            Strides::new(DynDim::from_array([1])),
            Strides::new(DynDim::from_array([1, 1])),
            Strides::new(DynDim::from_array([usize::MAX, 1])),
        ];
        let offsets = [0, 0, 1];

        for ((shape, strides), offset) in shapes.into_iter().zip(strides).zip(offsets) {
            assert!(Layout::new(shape, strides, offset).is_none());
        }
    }

    #[test]
    fn permuted_round_trip() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 5).expect("hand verified");
        let order = DimOrder::new(DynDim::from_array([1, 2, 0])).expect("hand verified");
        let permuted = layout.permuted(&order).unwrap();
        let restored = permuted
            .permuted(&order.inverse())
            .expect("same rank");

        assert_eq!(permuted.shape().as_slice(), &[3, 4, 2]);
        assert_eq!(permuted.strides().as_slice(), &[4, 1, 12]);
        assert_eq!(permuted.offset(), layout.offset());
        assert_eq!(permuted.max_offset(), layout.max_offset());
        assert_eq!(permuted.len(), layout.len());
        assert_eq!(restored.shape().as_slice(), layout.shape().as_slice());
        assert_eq!(restored.strides().as_slice(), layout.strides().as_slice());
    }

    #[test]
    fn memory_order_layouts() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let row = Layout::row_major(shape.clone(), 0).expect("hand verified");
        let column = Layout::column_major(shape, 0).expect("hand verified");

        assert_eq!(row.memory_order().as_slice(), &[0, 1, 2]);
        assert_eq!(column.memory_order().as_slice(), &[2, 1, 0]);

        let shape = Shape::new(DynDim::from_array([1, 1, 1]));
        let flat = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(flat.strides().as_slice(), &[1, 1, 1]);
        assert_eq!(flat.memory_order().as_slice(), &[0, 1, 2]);

        for layout in [row, column, flat] {
            assert!(DimOrder::new(DynDim::from_slice(layout.memory_order().as_slice())).is_some());
        }
    }

    #[test]
    fn packed_and_row_major() {
        let shape = Shape::new(DynDim::from_array([2, 3]));
        let row = Layout::row_major(shape, 0).expect("hand verified");
        let column_order = DimOrder::new(DynDim::from_array([1, 0])).unwrap();
        let column = row.permuted(&column_order).expect("same rank");

        assert!(row.is_packed() && row.is_row_major());
        assert_eq!(column.strides().as_slice(), &[1, 3]);
        assert!(column.is_packed());
        assert!(!column.is_row_major());

        let shape = Shape::new(DynDim::from_array([1, 7, 1]));
        let padded = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(padded.strides().as_slice(), &[7, 1, 1]);
        assert!(padded.is_packed() && padded.is_row_major());
    }

    #[test]
    fn fastest_and_contiguous_dimension() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(layout.fastest_dimension(), Some(DimIndex(2)));
        assert_eq!(layout.contiguous_dimension(), Some(DimIndex(2)));

        let shape = Shape::new(DynDim::from_array([1, 7, 1]));
        let padded = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(padded.fastest_dimension(), Some(DimIndex(1)));
        assert_eq!(padded.contiguous_dimension(), Some(DimIndex(1)));

        for extents in [&[1, 1][..], &[][..]] {
            let shape = Shape::new(DynDim::from_slice(extents));
            let layout = Layout::row_major(shape, 0).expect("hand verified");

            assert_eq!(layout.fastest_dimension(), None);
            assert_eq!(layout.contiguous_dimension(), None);
        }
    }

    #[test]
    fn lane_count_and_chunks() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(layout.lane_count(DimIndex(0)), Some(12));
        assert_eq!(layout.lane_count(DimIndex(1)), Some(8));
        assert_eq!(layout.lane_count(DimIndex(2)), Some(6));
        assert_eq!(layout.lane_count(DimIndex(3)), None);
        assert!(layout.lanes_are_chunks(DimIndex(2)));
        assert!(!layout.lanes_are_chunks(DimIndex(0)));
        assert!(!layout.lanes_are_chunks(DimIndex(3)));

        let shape = Shape::new(DynDim::from_array([3, 1, 2]));
        let padded = Layout::row_major(shape, 0).expect("hand verified");

        assert_eq!(padded.lane_count(DimIndex(1)), Some(6));
        assert!(padded.lanes_are_chunks(DimIndex(1)));
    }

    #[test]
    fn lanes_are_partitions() {
        let shape = Shape::new(DynDim::from_array([3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let order = layout.lexicographic_order();
        let rows = (0..layout.lane_count(DimIndex(1)).unwrap())
            .map(|lane| layout.lane_unvalidated(DimIndex(1), lane, &order))
            .collect::<Vec<LaneGeometry>>();
        let columns = (0..layout.lane_count(DimIndex(0)).unwrap())
            .map(|lane| layout.lane_unvalidated(DimIndex(0), lane, &order))
            .collect::<Vec<LaneGeometry>>();

        assert_eq!(
            rows.iter()
                .map(|g| g.offset())
                .collect::<Vec<usize>>(),
            vec![0, 4, 8]
        );
        assert!(
            rows.iter()
                .all(|g| g.stride() == 1 && g.len() == 4)
        );
        assert_eq!(
            columns
                .iter()
                .map(|g| g.offset())
                .collect::<Vec<usize>>(),
            vec![0, 1, 2, 3]
        );
        assert!(
            columns
                .iter()
                .all(|g| g.stride() == 4 && g.len() == 3)
        );

        for lanes in [rows, columns] {
            let mut offsets = lanes
                .iter()
                .flat_map(|g| g.offsets())
                .collect::<Vec<usize>>();
            offsets.sort_unstable();

            assert!(
                lanes
                    .iter()
                    .all(|g| g.max_offset().unwrap() <= layout.max_offset())
            );
            assert_eq!(offsets, (0..12).collect::<Vec<usize>>());
        }
    }

    #[test]
    fn lane_numbering_order() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let count = layout.lane_count(DimIndex(1)).unwrap();
        let offsets = |order: DimOrder<DynDim>| -> Vec<usize> {
            (0..count)
                .map(|lane| layout.lane_offset_unvalidated(DimIndex(1), lane, &order))
                .collect()
        };
        let mut lexicographic = offsets(layout.lexicographic_order());
        let mut colexicographic = offsets(DimOrder::colexicographic(3).unwrap());

        assert_eq!(count, 8);
        assert_eq!(lexicographic, vec![0, 1, 2, 3, 12, 13, 14, 15]);
        assert_eq!(colexicographic, vec![0, 12, 1, 13, 2, 14, 3, 15]);

        lexicographic.sort_unstable();
        colexicographic.sort_unstable();

        assert_eq!(lexicographic, colexicographic);
    }

    #[test]
    fn lane_selection() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([1, 2, 3]));
        let lane = layout.lane_at(DimIndex(1), &index).unwrap();

        assert_eq!((lane.offset(), lane.stride(), lane.len()), (15, 4, 3));
        assert_eq!(lane.offsets().collect::<Vec<_>>(), vec![15, 19, 23]);

        for at_dim in [0, 1, 2, 99] {
            let index = ArrayIndex::new(DynDim::from_array([1, at_dim, 3]));
            let other = layout.lane_at(DimIndex(1), &index).unwrap();

            assert_eq!(other, lane);
        }

        assert!(
            layout
                .lane_at(DimIndex(1), &ArrayIndex::new(DynDim::from_array([2, 0, 3])))
                .is_none()
        );
        assert!(
            layout
                .lane_at(DimIndex(1), &ArrayIndex::new(DynDim::from_array([1, 0, 4])))
                .is_none()
        );
        assert!(
            layout
                .lane_at(DimIndex(1), &ArrayIndex::new(DynDim::from_array([1, 0])))
                .is_none()
        );
        assert!(
            layout
                .lane_at(DimIndex(3), &ArrayIndex::new(DynDim::from_array([1, 0, 3])))
                .is_none()
        );
    }

    #[test]
    fn lane_geometry_predicates() {
        let shape = Shape::new(DynDim::from_array([2, 3, 4]));
        let layout = Layout::row_major(shape, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([1, 2, 0]));
        let contiguous = layout.lane_at(DimIndex(2), &index).unwrap();

        assert!(contiguous.is_contiguous() && contiguous.is_injective());
        assert_eq!(contiguous.contiguous_range(), Some(20..24));
        assert_eq!(contiguous.max_offset(), Some(23));
        assert!(contiguous.fits_within(24));
        assert!(!contiguous.fits_within(23));
        assert_eq!(contiguous.offset_of(2), Some(22));
        assert_eq!(contiguous.offset_of(3), Some(23));
        assert_eq!(contiguous.offset_of(4), None);

        let index = ArrayIndex::new(DynDim::from_array([0, 2, 3]));
        let strided = layout.lane_at(DimIndex(0), &index).unwrap();

        assert!(!strided.is_contiguous());
        assert_eq!(strided.contiguous_range(), None);
        assert_eq!(strided.offsets().collect::<Vec<_>>(), vec![11, 23]);

        let shape = Shape::new(DynDim::from_array([3, 1, 2]));
        let padded = Layout::row_major(shape, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([2, 0, 1]));
        let single = padded.lane_at(DimIndex(1), &index).unwrap();

        assert_eq!((single.offset(), single.stride(), single.len()), (5, 2, 1));
        assert!(single.is_contiguous());
        assert_eq!(single.contiguous_range(), Some(5..6));
    }

    #[test]
    fn non_overlapping_is_conservative() {
        let shape = Shape::new(DynDim::from_array([2, 2]));
        let strides = Strides::new(DynDim::from_array([2, 3]));
        let gapped = Layout::new(shape, strides, 0).expect("hand verified");
        let offsets = [[0, 0], [0, 1], [1, 0], [1, 1]]
            .into_iter()
            .map(|i| ArrayIndex::new(DynDim::from_array(i)))
            .map(|i| gapped.linear(&i).unwrap())
            .collect::<Vec<usize>>();

        assert_eq!(offsets, vec![0, 3, 2, 5]);
        assert!(!gapped.is_non_overlapping());

        let shapes = [
            Shape::new(DynDim::from_array([2, 2])),
            Shape::new(DynDim::from_array([3])),
            Shape::new(DynDim::from_array([1, 3])),
        ];
        let strides = [
            Strides::new(DynDim::from_array([1, 1])),
            Strides::new(DynDim::from_array([0])),
            Strides::new(DynDim::from_array([0, 1])),
        ];
        let expected = [false, false, true];

        for ((shape, strides), expected) in shapes.into_iter().zip(strides).zip(expected) {
            let layout = Layout::new(shape, strides, 0).expect("hand verified");

            assert_eq!(layout.is_non_overlapping(), expected);
        }
    }

    #[test]
    fn packed_with_degenerate_strides() {
        let shape = Shape::new(DynDim::from_array([2, 1, 3]));
        let strides = Strides::new(DynDim::from_array([3, 999, 1]));
        let layout = Layout::new(shape, strides, 0).expect("hand verified");

        assert_eq!(layout.max_offset(), 5);
        assert_eq!(layout.len(), 6);
        assert!(!layout.is_row_major());
        assert!(layout.is_packed());
        assert!(layout.is_non_overlapping());
    }

    #[test]
    fn tie_breaking_diverges() {
        let shape = Shape::new(DynDim::from_array([2, 2]));
        let strides = Strides::new(DynDim::from_array([1, 1]));
        let layout = Layout::new(shape, strides, 0).expect("hand verified");

        assert_eq!(layout.fastest_dimension(), Some(DimIndex(1)));
        assert_eq!(layout.contiguous_dimension(), Some(DimIndex(0)));
    }

    #[test]
    fn stride_zero_lane_is_not_injective() {
        let shape = Shape::new(DynDim::from_array([1, 3]));
        let strides = Strides::new(DynDim::from_array([0, 1]));
        let layout = Layout::new(shape, strides, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 1]));
        let lane = layout.lane_at(DimIndex(0), &index).unwrap();

        assert_eq!((lane.offset(), lane.stride(), lane.len()), (1, 0, 1));
        assert!(lane.is_injective());

        let shape = Shape::new(DynDim::from_array([3, 2]));
        let strides = Strides::new(DynDim::from_array([0, 1]));
        let broadcast = Layout::new(shape, strides, 0).expect("hand verified");
        let index = ArrayIndex::new(DynDim::from_array([0, 1]));
        let lane = broadcast.lane_at(DimIndex(0), &index).unwrap();

        assert_eq!((lane.offset(), lane.stride(), lane.len()), (1, 0, 3));
        assert!(!lane.is_injective());
        assert!(lane.fits_within(2));
    }
}
