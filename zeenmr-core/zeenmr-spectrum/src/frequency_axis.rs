//! Types for representing frequency axes.

mod axis;
pub use axis::{Axis, AxisGrid};

mod reference;
pub use reference::ShiftReference;

pub mod iter {
    //! Iterator types for axes.

    pub use crate::frequency_axis::axis::AxisIter;

    #[cfg(feature = "rayon")]
    pub use crate::frequency_axis::axis::ParAxisIter;
}
