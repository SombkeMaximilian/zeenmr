//! Types for representing frequency axes.

mod frequency_axis;
pub use frequency_axis::{FrequencyAxis, FrequencyGrid, Larmor, ShiftReference};

mod time_axis;
pub use time_axis::{TimeAxis, TimeGrid};

pub mod iter;
