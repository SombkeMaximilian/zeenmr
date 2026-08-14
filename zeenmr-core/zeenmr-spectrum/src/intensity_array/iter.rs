//! Array iterators.

mod indices;
pub use indices::Indices;
#[cfg(feature = "rayon")]
pub use indices::ParIndices;

mod lane_elements;
pub use lane_elements::{
    LaneElemContiguous, LaneElemContiguousMut, LaneElemStrided, LaneElemStridedMut, LaneOffsets,
};
#[cfg(feature = "rayon")]
pub use lane_elements::{
    ParLaneElemContiguous, ParLaneElemContiguousMut, ParLaneElemStrided, ParLaneElemStridedMut,
    ParLaneOffsets,
};

mod lanes;
pub use lanes::{LaneGeometries, Lanes, LanesMut};
#[cfg(feature = "rayon")]
pub use lanes::{ParLaneGeometries, ParLanes, ParLanesMut};

mod strided;
pub use strided::StridedIterKind;

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
