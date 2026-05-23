use crate::{Evaluate, PeakShape};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a [Gaussian] peak shape.
///
/// [Gaussian]: https://en.wikipedia.org/wiki/Gaussian_function
///
/// Also known as the normal distribution.
///
/// # Definition
///
/// The Gaussian function is typically defined as:
///
/// [Gaussian]: https://en.wikipedia.org/wiki/Gaussian_function
///
/// ```text
/// f(x) = a * exp(-(x - b)² / (2 c²))
/// ```
///
/// - `a` is the amplitude, maximum, or height of the peak (`amp`).
/// - `b` is the position of the maximum (`center`).
/// - `c` is the scale (`scale`).
///
/// Since `c` does not directly need to be absorbed, we can absorb both the
/// square and the factor into a single term, `double_scale2`, saving two
/// multiplications during every evaluation. The Gaussian function then becomes:
///
/// ```text
/// f(x) = amp * exp(-(x - center)² / double_scale2)
/// ```
///
/// This representation is used internally for efficiency, but the original
/// parameter can be recovered by inverting the transformation
///
/// ```text
/// scale = sqrt(double_scale2 / 2)
/// ```
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Gaussian` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gaussian {
    /// Amplitude of the Gaussian function.
    ///
    /// Must be positive.
    amp: f64,
    /// Doubled square of the standard deviation of the Gaussian function.
    ///
    /// Absorbed doubled, squared scale in the denominator of the exponential.
    /// Must be positive.
    double_scale2: f64,
    /// Center of the Gaussian function.
    center: f64,
}

impl Evaluate for Gaussian {
    fn evaluate(&self, at: f64) -> f64 {
        self.amp * (-(at - self.center).powi(2) / self.double_scale2).exp()
    }
}

impl PeakShape for Gaussian {
    fn center(&self) -> f64 {
        self.center
    }

    fn maximum(&self) -> f64 {
        self.amp
    }

    fn half_width(&self) -> f64 {
        (std::f64::consts::LN_2 * self.double_scale2).sqrt()
    }

    fn area(&self) -> f64 {
        self.amp * (std::f64::consts::PI * self.double_scale2).sqrt()
    }

    fn is_valid(&self) -> bool {
        self.amp.is_finite()
            && self.amp > 0.0
            && self.double_scale2.is_finite()
            && self.double_scale2 > 0.0
            && self.center.is_finite()
    }

    fn is_significant(&self, precision: f64) -> bool {
        self.maximum().abs() > precision && self.half_width().abs() > precision
    }
}

impl Gaussian {
    /// Creates a new `Gaussian` with the specified parameters.
    ///
    /// Note that these are not the standard parameters, but the transformed
    /// parameters as outlined in the struct documentation.
    pub fn new(amp: f64, double_scale2: f64, center: f64) -> Gaussian {
        Self {
            amp,
            double_scale2,
            center,
        }
    }

    /// Creates a new `Gaussian` from the untransformed parameters.
    pub fn from_untransformed(amp: f64, scale: f64, center: f64) -> Gaussian {
        Self {
            amp,
            double_scale2: 2_f64 * scale.powi(2),
            center,
        }
    }
}
