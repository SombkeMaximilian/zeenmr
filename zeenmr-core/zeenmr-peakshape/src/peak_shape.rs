use crate::Evaluate;

/// Trait that describes peak shapes in spectral data.
///
/// # Note for Implementors
///
/// While not technically required for a coherent definition of a structure
/// representing a peak shape, there are a few properties that should almost
/// always hold. In the following, let `f: S → ℝ` be our peak shape for some
/// set `S`:
///
/// - peak shapes are assumed to be symmetric about `center`, i.e., for all `d`,
///   it holds that `f(center - d) = f(center + d)`. Asymmetric shapes may
///   produce unexpected results in some algorithms.
/// - `center` should return the position at which the peak shape assumes its
///   maximum, as returned by `maximum`, i.e., `f(center) = maximum` exactly.
///   This property can be relaxed if the peak shape describes multiple peaks,
///   e.g., for splitting patterns in NMR spectra.
/// - `half_width` and `full_width` should be referring to the same width
///   property, e.g., the full/half width half maximum.
/// - `area` should correspond to the improper integral of `f` or some other
///   sensible definition of its area, irrespective of its position.
/// - `is_valid` should return `true` if and only if the peak shape is
///   geometrically sensible.
/// - `is_significant` should return `true` if and only if the internal
///   representation of its parameters is significant relative to the threshold.
///   This is to ensure numerical stability for its other operations. For
///   example, a peak shape with an area close to machine epsilon would likely
///   lead to downstream calculations producing nonsensical results.
pub trait PeakShape<T = f64>: Evaluate<T> {
    /// Returns the center position.
    fn center(&self) -> T;

    /// Returns the maximum value.
    fn maximum(&self) -> T {
        self.evaluate(self.center())
    }

    /// Returns the value of the half-width property.
    fn half_width(&self) -> T;

    /// Returns the value of the full-width property.
    fn full_width(&self) -> T;

    /// Returns the total area under the curve, as defined by the improper
    /// integral.
    fn area(&self) -> T;

    /// Returns `true` if the current values of the parameters represent a
    /// geometrically meaningful peak shape.
    fn is_valid(&self) -> bool;

    /// Returns `true` if any calculations are significant relative to the given
    /// precision.
    fn is_significant(&self, precision: T) -> bool;
}
