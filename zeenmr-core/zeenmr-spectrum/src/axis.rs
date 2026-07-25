//! Types for representing frequency axes.

mod frequency_axis;
pub use frequency_axis::{FrequencyAxis, FrequencyGrid};

mod reference;
pub use reference::ShiftReference;

pub mod iter {
    //! Iterator types for axes.

    pub use crate::axis::frequency_axis::AxisIter;

    #[cfg(feature = "rayon")]
    pub use crate::axis::frequency_axis::ParAxisIter;
}
