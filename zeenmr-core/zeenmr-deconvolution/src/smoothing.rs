//! Smoothing algorithms for the deconvolution process.

mod circular_buffer;
pub(crate) use circular_buffer::CircularBuffer;

mod identity;
pub use identity::Identity;

mod moving_average;
pub use moving_average::MovingAverage;

mod smooth;
pub use smooth::Smooth;
