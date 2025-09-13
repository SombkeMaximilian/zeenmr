//! Traits for estimating parameters of peak shape functions from data.

/// Trait for estimating parameters of a peak shape from three data points.
pub trait ThreePointStencil {
    /// Estimate the parameters of the peak shape from three data points.
    fn estimate_parameters(x: (f64, f64, f64), y: (f64, f64, f64)) -> Self;
}
