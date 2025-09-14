use crate::Evaluate;

/// Marker trait for types that can represent peak shapes in spectral data.
pub trait PeakShape: Evaluate + Center + Maximum + Width + Area + PrecisionCheck {}

impl<P> PeakShape for P where P: Evaluate + Center + Maximum + Width + Area + PrecisionCheck {}

/// Trait for the center of a peak shape.
pub trait Center {
    /// Returns the center position of the peak shape.
    fn center(&self) -> f64;
}

/// Trait for the maximum of a peak shape.
pub trait Maximum {
    /// Returns the maximum value of the peak shape.
    fn maximum(&self) -> f64;
}

/// Trait for the width of a peak shape.
///
/// This should be implemented with some reasonable width measure, e.g., full
/// width at half maximum.
pub trait Width {
    /// Returns the width of the peak shape.
    fn width(&self) -> f64;

    /// Returns half the width of the peak shape.
    fn half_width(&self) -> f64 {
        0.5 * self.width()
    }
}

/// Trait for the area under a peak shape curve.
pub trait Area {
    /// Returns the area under the peak shape curve.
    fn area(&self) -> f64;
}

/// Trait for checking if a peak shape's parameters are below a certain
/// precision threshold.
///
/// Some algorithms may produce peak shapes with parameters whose absolute
/// value is nearly zero due to artifacts in the data or numerical issues.
/// This trait provides a method to filter out such artifacts.
///
/// # For Implementors
///
/// This trait should **not** check if a position parameter is below the
/// precision, as positions can validly be near zero.
pub trait PrecisionCheck {
    /// Checks if any of the peak shape's parameters are below the given
    /// precision.
    fn parameters_below(&self, precision: f64) -> bool;

    /// Checks if all the peak shape's parameters are above the given precision.
    fn parameters_above(&self, precision: f64) -> bool {
        !self.parameters_below(precision)
    }
}
