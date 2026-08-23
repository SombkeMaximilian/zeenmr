//! Types for representing frequency axes.

pub mod iter;

pub mod range;

mod axes;
pub use axes::{Axes, DynFrequencyAxes, DynTimeAxes, StaticFrequencyAxes, StaticTimeAxes, frequency_axes, time_axes};

mod frequency_axis;
pub use frequency_axis::{FrequencyAxis, FrequencyGrid, Larmor, ShiftReference};

mod time_axis;
pub use time_axis::{TimeAxis, TimeGrid};
