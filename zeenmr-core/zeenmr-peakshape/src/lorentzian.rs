use crate::{Evaluate, EvaluateParts, FromArray, PeakShape};
use num_traits::{Float, FloatConst};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a [Lorentzian] peak shape.
///
/// [Lorentzian]: https://en.wikipedia.org/wiki/Cauchy_distribution
///
/// Also known as the probability density function of the Cauchy distribution,
/// the Lorentz distribution, or the Breit-Wigner distribution.
///
/// This type implements the [`Evaluate`] trait and is a [`PeakShape`], if its
/// type parameter is [`Float`] + [`FloatConst`].
///
/// # Definition
///
/// The [Lorentzian] function is typically defined as:
///
/// [Lorentzian]: https://en.wikipedia.org/wiki/Cauchy_distribution
///
/// ```text
/// f(x) = 1/π * γ / ((x - x₀)² + γ²)
/// ```
///
/// - `γ` is the scale or half-width at half-maximum (`scale`).
/// - `x₀` is the position of the maximum (`center`).
/// - The factor `1/π` ensures the integral of the function equals 1.
///
/// To fit the function to data, an amplitude parameter `amp` introduced, which
/// is proportional to the height of the peak and replaces `1/π`, resulting in:
///
/// ```text
/// f(x) = amp * scale / (scale² + (x - center)²)
/// ```
///
/// This form is computationally inefficient, as both the numerator and
/// scale² in the denominator are products of constants. They can be absorbed
/// into single parameters with the following transformation:
///
/// ```text
/// amp_scale = amp * scale
/// scale2 = scale²
/// ```
///
/// The Lorentzian function then becomes:
///
/// ```text
/// f(x) = amp_scale / (scale2 + (x - center)²)
/// ```
///
/// This representation is used internally for efficiency, but the original
/// parameters can be recovered by inverting the transformations:
///
/// ```text
/// amp = amp_scale / sqrt(scale2)
/// scale = sqrt(scale2)
/// ```
///
/// # Negative Parameters
///
/// For a `Lorentzian` to represent a peak shape, its `amp` and `scale` must
/// be positive. This is not enforced to avoid unnecessary overhead.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Lorentzian` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lorentzian<T> {
    /// Numerator of the Lorentzian function.
    ///
    /// Absorbed product of amplitude parameter and scale. Proportional to the
    /// height of the peak. Must be positive.
    amp_scale: T,
    /// Scale parameter, or half-width at half-maximum (HWHM), squared.
    ///
    /// Absorbed squared scale in the denominator of the Lorentzian function.
    /// Must be positive.
    scale2: T,
    /// Center of the Lorentzian function.
    center: T,
}

impl<T> FromArray<T, 3> for Lorentzian<T>
where
    T: Float,
{
    /// Creates a new `Lorentzian` from its parameter array.
    ///
    /// Order: `amp_scale`, `scale2`, `center`.
    fn from_array(array: [T; 3]) -> Self {
        Self::new(array[0], array[1], array[2])
    }
}

impl<T> Evaluate<T> for Lorentzian<T>
where
    T: Float,
{
    fn evaluate(&self, at: T) -> T {
        self.amp_scale / (self.scale2 + (at - self.center).powi(2))
    }
}

impl<T> EvaluateParts<T> for Lorentzian<T>
where
    T: Float,
{
    fn parts(&self, at: T) -> (T, T) {
        let den = self.scale2 + (at - self.center).powi(2);

        (self.amp_scale, den)
    }

    fn num_bounds(&self, _: T, _: T) -> (T, T) {
        (self.amp_scale, self.amp_scale)
    }

    fn den_bounds(&self, lo: T, hi: T) -> (T, T) {
        let near = self.center - self.center.max(lo).min(hi);
        let den_lo = (lo - self.center).powi(2);
        let den_hi = (hi - self.center).powi(2);

        (self.scale2 + near.powi(2), self.scale2 + den_lo.max(den_hi))
    }
}

impl<T> PeakShape<T> for Lorentzian<T>
where
    T: Float + FloatConst,
{
    fn center(&self) -> T {
        self.center
    }

    fn maximum(&self) -> T {
        self.amp_scale / self.scale2
    }

    fn half_width(&self) -> T {
        self.scale2.sqrt()
    }

    fn full_width(&self) -> T {
        (T::one() + T::one()) * self.half_width()
    }

    fn area(&self) -> T {
        T::PI() * self.amp_scale / self.scale2.sqrt()
    }

    fn is_valid(&self) -> bool {
        self.amp_scale.is_finite()
            && self.amp_scale > T::zero()
            && self.scale2.is_finite()
            && self.scale2 > T::zero()
            && self.center.is_finite()
    }

    fn is_significant(&self, precision: T) -> bool {
        self.maximum().abs() > precision && self.scale2.abs() > precision
    }
}

impl<T> Lorentzian<T>
where
    T: Float,
{
    /// Creates a new `Lorentzian` with the specified parameters.
    ///
    /// Note that these are not the standard parameters, but the transformed
    /// parameters as outlined in the struct documentation.
    pub fn new(amp_scale: T, scale2: T, center: T) -> Self {
        Self {
            amp_scale,
            scale2,
            center,
        }
    }

    /// Creates a new `Lorentzian` from the untransformed parameters.
    pub fn from_untransformed(amp: T, scale: T, center: T) -> Self {
        Self {
            amp_scale: amp * scale,
            scale2: scale.powi(2),
            center,
        }
    }
}
