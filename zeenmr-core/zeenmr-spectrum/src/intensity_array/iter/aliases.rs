use crate::intensity_array::iter::{
    ArrayElemContiguous, ArrayElemContiguousMut, ArrayElemStrided, ArrayElemStridedMut,
    LaneElemStrided, LaneElemStridedMut, StridedIterKind,
};

#[cfg(feature = "rayon")]
use crate::intensity_array::iter::{Indices, LaneGeometries, LaneOffsets, Lanes, LanesMut};
#[cfg(feature = "rayon")]
use crate::iter::Par;

/// Parallel iterator over the multidimensional indices of a shape.
///
/// Yields indices in lexicographic order: the last dimension varies fastest.
#[cfg(feature = "rayon")]
pub type ParIndices<D> = Par<Indices<D>>;

/// Parallel, contiguous iterator over the elements of an array.
#[cfg(feature = "rayon")]
pub type ParArrayElemContiguous<'s, T, D> = Par<ArrayElemContiguous<'s, T, D>>;

/// Parallel, contiguous iterator over mutable references to the elements of an
/// array.
#[cfg(feature = "rayon")]
pub type ParArrayElemContiguousMut<'s, T, D> = Par<ArrayElemContiguousMut<'s, T, D>>;

/// Parallel, strided iterator over the elements of an array.
#[cfg(feature = "rayon")]
pub type ParArrayElemStrided<'s, T, D> = Par<ArrayElemStrided<'s, T, D>>;

/// Parallel, strided iterator over mutable references to the elements of an
/// array.
#[cfg(feature = "rayon")]
pub type ParArrayElemStridedMut<'s, T, D> = Par<ArrayElemStridedMut<'s, T, D>>;

/// Iterator over the elements of an array.
pub type ArrayElem<'s, T, D> =
    StridedIterKind<ArrayElemContiguous<'s, T, D>, ArrayElemStrided<'s, T, D>>;

/// Parallel iterator over the elements of an array.
#[cfg(feature = "rayon")]
pub type ParArrayElem<'s, T, D> =
    StridedIterKind<ParArrayElemContiguous<'s, T, D>, ParArrayElemStrided<'s, T, D>>;

/// Iterator over mutable reference to the elements of an array.
pub type ArrayElemMut<'s, T, D> =
    StridedIterKind<ArrayElemContiguousMut<'s, T, D>, ArrayElemStridedMut<'s, T, D>>;

/// Parallel iterator over mutable reference to the elements of an array.
#[cfg(feature = "rayon")]
pub type ParArrayElemMut<'s, T, D> =
    StridedIterKind<ParArrayElemContiguousMut<'s, T, D>, ParArrayElemStridedMut<'s, T, D>>;

/// Parallel iterator over the buffer offsets of a lane along one dimension.
#[cfg(feature = "rayon")]
pub type ParLaneOffsets = Par<LaneOffsets>;

/// Parallel, strided iterator over the elements of a buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemStrided<'s, T> = Par<LaneElemStrided<'s, T>>;

/// Parallel, strided iterator over mutable references of the elements of a
/// buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemStridedMut<'s, T> = Par<LaneElemStridedMut<'s, T>>;

/// Iterator over contiguous elements of a buffer.
pub type LaneElemContiguous<'s, T> = std::slice::Iter<'s, T>;

/// Parallel iterator over contiguous elements of a buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguous<'s, T> = rayon::slice::Iter<'s, T>;

/// Iterator over mutable references of contiguous elements of a buffer.
pub type LaneElemContiguousMut<'s, T> = std::slice::IterMut<'s, T>;

/// Parallel iterator over mutable references of contiguous elements of a
/// buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguousMut<'s, T> = rayon::slice::IterMut<'s, T>;

/// Iterators over the elements of a lane view.
pub type LaneElem<'s, T> = StridedIterKind<LaneElemContiguous<'s, T>, LaneElemStrided<'s, T>>;

/// Parallel iterators over the elements of a lane view.
#[cfg(feature = "rayon")]
pub type ParLaneElem<'s, T> =
    StridedIterKind<ParLaneElemContiguous<'s, T>, ParLaneElemStrided<'s, T>>;

/// Iterator over the elements of a mutable lane view.
pub type LaneElemMut<'s, T> =
    StridedIterKind<LaneElemContiguousMut<'s, T>, LaneElemStridedMut<'s, T>>;

/// Parallel iterator over the elements of a mutable lane view.
#[cfg(feature = "rayon")]
pub type ParLaneElemMut<'s, T> =
    StridedIterKind<ParLaneElemContiguousMut<'s, T>, ParLaneElemStridedMut<'s, T>>;

/// Parallel iterator over the lanes of a layout along one dimension.
#[cfg(feature = "rayon")]
pub type ParLaneGeometries<D> = Par<LaneGeometries<D>>;

/// Parallel iterator over the lanes of a layout along one dimension.
#[cfg(feature = "rayon")]
pub type ParLanes<'s, T, D> = Par<Lanes<'s, T, D>>;

/// Parallel iterator over mutable lanes of an array along one dimension.
#[cfg(feature = "rayon")]
pub type ParLanesMut<'s, T, D> = Par<LanesMut<'s, T, D>>;
