//! Types for representing frequency axes.

mod frequency_axis;
pub use frequency_axis::{FrequencyAxis, FrequencyGrid};

mod time_axis;
pub use time_axis::{TimeAxis, TimeGrid};

mod reference;
pub use reference::ShiftReference;

pub mod iter;
