//! Array iterators.

mod indices;
pub use indices::Indices;

mod lane_elements;
pub use lane_elements::{LaneElemStrided, LaneElemStridedMut, LaneOffsets};

mod lanes;
pub use lanes::{LaneGeometries, Lanes, LanesMut};

mod strided;
pub use strided::StridedIterKind;

#[cfg(feature = "rayon")]
mod par;
#[cfg(feature = "rayon")]
pub use par::Par;

/// Iterators that can be split into two halves at an index.
///
/// # Safety
///
/// `split_at` must yield at most `index` items on the left and the remainder,
/// if any, on the right, in the original order. Left and right must be
/// disjoint at the boundary in terms of yielded element enumeration. That is,
/// if we split at index `k`, left must only contain elements `[start, k)` and
/// right must only contain elements `[k, len)`.
pub unsafe trait SplitAt: Sized {
    /// Splits the iterator at the provided `index`.
    fn split_at(self, index: usize) -> (Self, Self);
}

/// Parallel iterator over the multidimensional indices of a shape.
///
/// Yields indices in lexicographic order: the last dimension varies fastest.
pub type ParIndices<D> = Par<Indices<D>>;

/// Parallel iterator over the buffer offsets of a lane along one dimension.
pub type ParLaneOffsets = Par<LaneOffsets>;

/// Parallel, strided iterator over the elements of a buffer.
pub type ParLaneElemStrided<'s, T> = Par<LaneElemStrided<'s, T>>;

/// Parallel, strided iterator over mutable references of the elements of a
/// buffer.
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
pub type ParLaneGeometries<D> = Par<LaneGeometries<D>>;

/// Parallel iterator over the lanes of a layout along one dimension.
pub type ParLanes<'s, T, D> = Par<Lanes<'s, T, D>>;

/// Parallel iterator over mutable lanes of an array along one dimension.
pub type ParLanesMut<'s, T, D> = Par<LanesMut<'s, T, D>>;
