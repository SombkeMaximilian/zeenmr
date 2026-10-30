//! Types for representing frequency axes.

pub mod iter;

pub mod range;

mod axes;
pub use axes::{
    Axes, Axis, DiscreteAxes, DynFrequencyAxes, DynTimeAxes, StaticFrequencyAxes, StaticTimeAxes,
    frequency_axes, time_axes,
};

mod frequency_axis;
pub use frequency_axis::{DiscreteFrequencyAxis, FrequencyAxis, Larmor, ShiftReference};

mod time_axis;
pub use time_axis::{DiscreteTimeAxis, TimeAxis};
