use crate::{Evaluate, PeakShape};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a [Lorentzian] peak shape.
///
/// [Lorentzian]: https://en.wikipedia.org/wiki/Cauchy_distribution
///
/// Also known as the probability density function of the Cauchy distribution,
/// the Lorentz distribution, or the Breit-Wigner distribution.
///
/// This type implements the [`Evaluate`] trait, and is a [`PeakShape`].
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
/// For a [`Lorentzian`] to represent a peak shape, its `amp` and `scale` must
/// be positive. Further, if the transformed parameters `amp_scale` and
/// `scale2` are negative, attempting to recover the `scale` parameter will
/// corrupt the data, as it involves taking the square root. This is not
/// enforced to avoid unnecessary overhead, and any algorithms using this type
/// must be designed to prevent negative parameters.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Lorentzian` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lorentzian {
    /// Numerator of the Lorentzian function.
    ///
    /// Absorbed product of amplitude parameter and scale. Proportional to the
    /// height of the peak. Must be positive.
    amp_scale: f64,
    /// Scale parameter, or half-width at half-maximum (HWHM), squared.
    ///
    /// Absorbed squared scale in the denominator of the Lorentzian function.
    /// Must be positive.
    scale2: f64,
    /// Center of the Lorentzian function.
    center: f64,
}

impl Evaluate for Lorentzian {
    fn evaluate(&self, at: f64) -> f64 {
        self.amp_scale / (self.scale2 + (at - self.center).powi(2))
    }
}

impl PeakShape for Lorentzian {
    fn center(&self) -> f64 {
        self.center
    }

    fn maximum(&self) -> f64 {
        self.amp_scale / self.scale2
    }

    fn half_width(&self) -> f64 {
        self.scale2.sqrt()
    }

    fn full_width(&self) -> f64 {
        2.0 * self.scale2.sqrt()
    }

    fn area(&self) -> f64 {
        std::f64::consts::PI * self.amp_scale / self.scale2.sqrt()
    }

    fn integral(&self, from: f64, to: f64) -> f64 {
        let scale = self.scale2.sqrt();

        (self.amp_scale / scale) * (((self.center - from) / scale).atan() - ((self.center - to) / scale).atan())
    }

    fn is_valid(&self) -> bool {
        self.amp_scale.is_finite() && self.amp_scale > 0.0
            && self.scale2.is_finite() && self.scale2 > 0.0
            && self.center.is_finite()
    }

    fn is_significant(&self, precision: f64) -> bool {
        self.maximum().abs() > precision && self.scale2.abs() > precision
    }
}

impl Lorentzian {
    /// Creates a new `Lorentzian` with the specified parameters.
    pub fn new(amp: f64, scale: f64, center: f64) -> Self {
        Self {
            amp_scale: amp * scale,
            scale2: scale.powi(2),
            center,
        }
    }
}
