//! Smoothing algorithms for the deconvolution process.

mod coefficient_window;
pub use coefficient_window::{CoefficientWindow, CorrectEdges, MirrorCorrection};

mod moving_average;
pub use moving_average::MovingAverage;

mod least_squares;
pub use least_squares::LeastSquares;

mod smooth;
pub use smooth::{Chain, Identity, Smooth};
