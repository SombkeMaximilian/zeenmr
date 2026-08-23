//! Array iterators.

mod aliases;
pub use aliases::*;

mod indices;
pub use indices::Indices;

mod lane_elements;
pub use lane_elements::{LaneElemStrided, LaneElemStridedMut, LaneOffsets};

mod lanes;
pub use lanes::{LaneGeometries, Lanes, LanesMut};

mod strided;
pub use strided::StridedIterKind;
