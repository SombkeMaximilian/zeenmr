//! Smoothing algorithms for the deconvolution process.

mod identity;
pub use identity::Identity;

mod moving_average;
pub use moving_average::MovingAverage;

mod smooth;
pub use smooth::Smooth;
