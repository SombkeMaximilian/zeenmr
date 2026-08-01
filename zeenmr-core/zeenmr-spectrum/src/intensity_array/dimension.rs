use crate::intensity_array::iter::{IndicesRowMajor, LanesRowMajor};
use std::ops::{Deref, DerefMut};

#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{ParIndicesRowMajor, ParLanesRowMajor};

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

/// Multidimensional quantity representation with a size determined at runtime.
#[derive(Clone, Debug)]
enum DynDimInner {
    /// On the stack for up to [`MAX_INLINE_RANK`].
    Stack(StaticLen, [usize; MAX_INLINE_RANK]),
    /// On the heap for higher dimensions.
    Heap(Vec<usize>),
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

/// Multidimensional index into an array.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct ArrayIndex<D>(D);

impl<D> ArrayIndex<D>
where
    D: Dimension,
{
    /// Creates a new multidimensional array index.
    pub fn new(dim: D) -> Self {
        Self(dim)
    }

    /// Returns the `linear`-th index of `shape` in row-major order.
    ///
    /// `linear` must be less than the product of `shape`'s extents. Otherwise,
    /// the leading component of the result exceeds its extent.
    pub(crate) fn linear_in_shape(linear: usize, shape: &Shape<D>) -> Self {
        let mut index =
            Self::new(D::zero(shape.rank()).expect("`D` can always represent its own rank"));
        index.set_linear_row_major(linear, shape.as_slice());

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

    /// Increments the index according to the `extents` in row-major order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    ///
    /// Returns `false` if the index was the last one, in which case every
    /// component wraps to zero. This enables `while` loops to be written in the
    /// following way:
    ///
    /// `while { ...; index.increment_row_major(extents) } {}`
    ///
    /// though callers must not rely on it terminating when any extent is zero.
    /// No component can then reach its extent, so the index never advances past
    /// it. Drive such loops with an external count instead.
    pub(crate) fn increment_row_major(&mut self, extents: &[usize]) -> bool {
        debug_assert_eq!(self.rank(), extents.len());

        let components = self.0.as_mut_slice();
        for (component, &extent) in components.iter_mut().zip(extents).rev() {
            *component += 1;
            if *component < extent {
                return true;
            }
            *component = 0;
        }

        false
    }

    /// Reverses the index according to the `extents` in row-major order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    ///
    /// Returns `false` if the index was the first one, in which case every
    /// component wraps to the maximum for the given extents. This enables
    /// `while` loops to be written in the following way:
    ///
    /// `while { ...; index.decrement_row_major(extents) } {}`
    ///
    /// though callers must not rely on it terminating when any extent is zero.
    /// No component can then reach its extent, so the index never advances past
    /// it. Drive such loops with an external count instead.
    pub(crate) fn decrement_row_major(&mut self, extents: &[usize]) -> bool {
        debug_assert_eq!(self.rank(), extents.len());

        let components = self.0.as_mut_slice();
        for (component, &extent) in components.iter_mut().zip(extents).rev() {
            if *component > 0 {
                *component -= 1;
                return true;
            }
            *component = extent.saturating_sub(1);
        }

        false
    }

    /// Sets the index to the `k`-th one in row-major order.
    ///
    /// `self` and `extents` must have the same rank, and every component of
    /// `self` must be less than the corresponding extent.
    pub(crate) fn set_linear_row_major(&mut self, mut linear: usize, extents: &[usize]) {
        debug_assert_eq!(self.rank(), extents.len());
        debug_assert!(
            extents
                .iter()
                .try_fold(1usize, |a, &e| a.checked_mul(e))
                .is_none_or(|p| linear < p)
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

/// Shape of an array.
///
/// The entries represent the extent of the array along each dimension.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Shape<D>(D);

impl<D> Shape<D>
where
    D: Dimension,
{
    /// Creates a new array shape.
    pub fn new(dim: D) -> Self {
        Self(dim)
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

    /// Returns an iterator over the multidimensional indices in row-major
    /// order.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    pub fn indices_row_major(&self) -> Option<IndicesRowMajor<D>> {
        IndicesRowMajor::new(self.clone())
    }

    /// Returns a parallel iterator over the multidimensional indices in
    /// row-major order.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    #[cfg(feature = "rayon")]
    pub fn par_indices_row_major(&self) -> Option<ParIndicesRowMajor<D>> {
        ParIndicesRowMajor::new(self.clone())
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
    shape: Shape<D>,
    /// Array strides.
    strides: Strides<D>,
    /// Offset from the start of the buffer.
    offset: usize,
    /// Element count of the layout.
    len: usize,
}

impl<D> Layout<D>
where
    D: Dimension,
{
    /// Creates a row-major, contiguous layout with the given shape and offset.
    ///
    /// Returns `None` if [`Shape::product_checked`] returns `Some(0)` or `None`
    /// or if no contiguous strides can be computed from `shape`.
    pub fn row_major(shape: Shape<D>, offset: usize) -> Option<Self> {
        let len = shape
            .product_checked()
            .filter(|&size| size != 0)?;
        let strides = shape.row_major_strides()?;

        Some(Self {
            shape,
            strides,
            offset,
            len,
        })
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
    /// This always returns `false` for contiguous layouts.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Largest buffer offset this layout can address.
    ///
    /// Returns `None` if overflow occurs.
    pub fn max_offset(&self) -> Option<usize> {
        self.shape
            .0
            .as_slice()
            .iter()
            .zip(self.strides.0.as_slice())
            .try_fold(self.offset, |acc, (&extent, &stride)| {
                acc.checked_add((extent - 1).checked_mul(stride)?)
            })
    }

    /// Returns `true` if the layout covers `offset..offset + len` with no gaps.
    ///
    /// Dimensions with an extent of 1 are ignored, since their stride is never
    /// used to address an element.
    pub fn is_packed(&self) -> bool {
        let extents = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let order = self.iteration_order();

        let mut expected = 1_usize;
        for (extent, stride) in order
            .as_slice()
            .iter()
            .rev()
            .map(|&dim| (extents[dim], strides[dim]))
        {
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

    /// Returns the dimension with the smallest stride.
    ///
    /// Dimensions with an extent of 1 are ignored, since their stride never
    /// addresses an element. Ties are broken toward the later dimension, which
    /// matches row-major order. Returns `None` if every extent is 1, including
    /// for rank 0.
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

    /// Returns the dimension indices ordered from largest to smallest stride.
    ///
    /// Walking dimensions in this order visits the buffer as sequentially as
    /// the layout permits. Ties are broken by ascending dimension index, so a
    /// row-major layout yields `0, 1, ..., rank - 1`.
    pub fn iteration_order(&self) -> D {
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

        order
    }

    /// Returns an iterator over the lanes in row-major order.
    ///
    /// Returns `None` if `dim` is out of range.
    pub fn lanes_row_major(&self, dim: DimIndex) -> Option<LanesRowMajor<D>> {
        LanesRowMajor::new(self.clone(), dim)
    }

    /// Returns a parallel iterator over the lanes in row-major order.
    ///
    /// Returns `None` if `dim` is out of range.
    #[cfg(feature = "rayon")]
    pub fn par_lanes_row_major(&self, dim: DimIndex) -> Option<ParLanesRowMajor<D>> {
        ParLanesRowMajor::new(self.clone(), dim)
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

    /// Returns the geometry of the `lane`-th lane along `dim`.
    ///
    /// `dim` must be in range and `lane` must be less than
    /// [`lane_count`](Self::lane_count). Otherwise, the result is meaningless.
    pub(crate) fn lane_row_major_unvalidated(&self, dim: DimIndex, lane: usize) -> Lane {
        debug_assert!(dim.0 < self.shape.rank());
        debug_assert!(self.lane_count(dim).is_some_and(|count| lane < count));

        Lane {
            offset: self.lane_offset_row_major_unvalidated(dim, lane),
            stride: self.strides.as_slice()[dim.0],
            count: self.shape.as_slice()[dim.0],
        }
    }

    /// Returns the buffer offset of the first element of the `lane`-th lane
    /// along `dim`, numbered in row-major order over the other dimensions.
    ///
    /// `dim` must be in range and `lane` must be less than
    /// [`lane_count`](Self::lane_count). Otherwise, the result is meaningless.
    ///
    /// No arithmetic can overflow under those preconditions: every offset this
    /// produces is at most [`Layout::max_offset`], which was validated at
    /// construction.
    pub(crate) fn lane_offset_row_major_unvalidated(
        &self,
        dim: DimIndex,
        mut lane: usize,
    ) -> usize {
        debug_assert!(dim.0 < self.shape.rank());
        debug_assert!(self.lane_count(dim).is_some_and(|count| lane < count));

        let extents = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let dim = dim.0;

        extents
            .iter()
            .zip(strides)
            .enumerate()
            .rev()
            .filter(|&(i, _)| i != dim)
            .fold(self.offset, |acc, (_, (&extent, &stride))| {
                let acc = acc + (lane % extent) * stride;
                lane /= extent;

                acc
            })
    }
}

impl<D1> Layout<D1>
where
    D1: Dimension,
{
    /// Returns the linear buffer offset of `index`.
    ///
    /// Returns `None` if `index` has a different rank than the layout, if any
    /// component is out of bounds, or if the offset overflows.
    pub fn linear<D2>(&self, index: &ArrayIndex<D2>) -> Option<usize>
    where
        D2: Dimension,
    {
        if let (Some(self_rank), Some(index_rank)) = (D1::RANK, D2::RANK) {
            if self_rank != index_rank {
                return None;
            }
        } else if index.rank() != self.shape.rank() {
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
    /// In particular, it must hold that
    /// - `index` and layout have the same rank, and
    /// - none of the components of `index` is out of bounds, and
    /// - the offset doesn't overflow for this index.
    ///
    /// If any of these conditions do not hold, the result is meaningless.
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
}

/// Geometry of a single lane through an array buffer.
///
/// Describes the `count` buffer offsets `offset`, `offset + stride`, ...,
/// `offset + (count - 1) * stride`. Carries no dimension information: a lane
/// is a flat walk, and the layout it came from determines what it traverses.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Lane {
    /// Offset of the first element.
    pub offset: usize,
    /// Number of elements between consecutive elements of the lane.
    pub stride: usize,
    /// Number of elements in the lane.
    pub count: usize,
}

impl Lane {
    /// Returns the buffer offset of the `index`-th element.
    #[inline]
    pub fn offset_of(&self, index: usize) -> usize {
        debug_assert!(index < self.count);

        self.offset + index * self.stride
    }

    /// Returns `true` if the lane's elements are adjacent in the buffer.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == 1 || self.count <= 1
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
    fn shape() {
        for (extents, _, len) in CONTIGUOUS {
            let shape = Shape::new(DynDim::from_slice(extents));

            assert_eq!(shape.product_checked(), Some(len));
        }
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
    fn contiguous_layout() {
        for (extents, _, len) in CONTIGUOUS {
            let shape = Shape::new(DynDim::from_slice(extents));

            for offset in [0, 1, 1000] {
                let layout = Layout::row_major(shape.clone(), offset).expect("hand verified");

                assert_eq!(layout.len(), len);
                assert_eq!(layout.max_offset(), Some(offset + len - 1));
            }
        }
    }

    #[test]
    fn rank_zero_scalar() {
        let layout =
            Layout::row_major(Shape::new(DynDim::from_slice(&[][..])), 9).expect("hand verified");

        assert_eq!(layout.len(), 1);
        assert_eq!(layout.shape().as_slice(), &[] as &[usize]);
        assert_eq!(layout.max_offset(), Some(9));
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
        let layout = Layout::row_major(shape, usize::MAX - 2).expect("hand verified");

        assert!(layout.max_offset().is_none());
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
    fn zero_respects_rank() {
        assert!(StaticDim::<3>::zero(3).is_some());
        assert!(StaticDim::<3>::zero(2).is_none());
        assert!(StaticDim::<3>::zero(1).is_none());
        assert!(StaticDim::<3>::zero(0).is_none());
        assert_eq!(StaticDim::<3>::zero(3).unwrap().as_slice(), &[0; 3]);
        assert_eq!(DynDim::zero(9).unwrap().as_slice(), &[0; 9]);
    }
}
