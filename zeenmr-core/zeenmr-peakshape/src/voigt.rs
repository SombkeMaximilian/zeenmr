use crate::{Evaluate, PeakShape};
use num_traits::{Float, FloatConst};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a [pseudo-Voigt] peak shape.
///
/// [pseudo-Voigt]: https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
///
/// A linear combination of a Lorentzian and a Gaussian component that share the
/// same center and the same half-width at half-maximum (HWHM), used as a
/// computationally cheap approximation to the true Voigt profile (which
/// requires a convolution).
///
/// This type implements the [`Evaluate`] trait and is a [`PeakShape`], if its
/// type parameter is [`Float`] + [`FloatConst`].
///
/// # Definition
///
/// The [pseudo-Voigt] function is defined as:
///
/// [pseudo-Voigt]: https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
///
/// ```text
/// f(x) = amp * [η * L(x) + (1 - η) * G(x)]
/// ```
///
/// where `L` and `G` are the Lorentzian and Gaussian components, each
/// normalized to a maximum value of 1 at `x = x₀`:
///
/// ```text
/// L(x) = 1 / (1 + ((x - x₀) / γ)²)
/// G(x) = exp(-ln(2) * ((x - x₀) / γ)²)
/// ```
///
/// - `x₀` is the position of the maximum (`center`).
/// - `γ` is the shared half-width at half-maximum (`scale`).
/// - `η` (`eta`) is the mixing parameter, `0 ≤ η ≤ 1`. `η = 1` recovers a pure
///   Lorentzian, `η = 0` a pure Gaussian.
/// - `amp` is the peak height, i.e. `f(x₀) = amp`.
///
/// Because both components are normalized to a maximum of 1, `amp` is the
/// actual peak height directly, unlike [`Lorentzian`](crate::Lorentzian)'s
/// `amp_scale`, which requires inverting a product to recover the height.
///
/// As with the Lorentzian implementation, `γ` is stored pre-squared as `scale2`
/// to avoid repeated `sqrt`/`powi` round-trips during evaluation.
///
/// # Shared-Width Approximation
///
/// Tying the Gaussian and Lorentzian components to a single width parameter is
/// the standard pseudo-Voigt simplification. The true FWHM of the summed curve
/// is not exactly `2γ`; `half_width`/`full_width` below report the shared
/// nominal parameter, not a re-derived FWHM of the mixed curve. This is the
/// conventional convention used in most peak-fitting literature.
///
/// # Negative and Out-of-Range Parameters
///
/// For a `Voigt` to represent a peak shape, `amp` and `scale2` must be
/// positive, and `eta` must lie in `[0, 1]`. This is not enforced at
/// construction to avoid unnecessary overhead.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Voigt` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Voigt<T> {
    /// Peak height.
    ///
    /// Must be positive.
    amp: T,
    /// Shared half-width at half-maximum (HWHM), squared.
    ///
    /// Absorbed square. Must be positive.
    scale2: T,
    /// Center of the pseudo-Voigt function.
    center: T,
    /// Lorentzian mixing fraction.
    ///
    /// Must lie in `[0, 1]`.
    eta: T,
}

impl<T> Evaluate<T> for Voigt<T>
where
    T: Float + FloatConst,
{
    fn evaluate(&self, at: T) -> T {
        let center_dist_2 = (at - self.center).powi(2);
        let lorentzian = self.scale2 / (self.scale2 + center_dist_2);
        let gaussian = (-T::LN_2() * center_dist_2 / self.scale2).exp();

        self.amp * (self.eta * lorentzian + (T::one() - self.eta) * gaussian)
    }
}

impl<T> PeakShape<T> for Voigt<T>
where
    T: Float + FloatConst,
{
    fn center(&self) -> T {
        self.center
    }

    fn maximum(&self) -> T {
        self.amp
    }

    fn half_width(&self) -> T {
        self.scale2.sqrt()
    }

    fn full_width(&self) -> T {
        T::from(2_u8)
    }

    fn area(&self) -> T {
        let half_width = self.half_width();
        let lorentzian_area = T::PI() * half_width;
        let gaussian_area = half_width * (T::PI() / T::LN_2()).sqrt();

        self.amp * (self.eta * lorentzian_area + (T::one() - self.eta) * gaussian_area)
    }

    fn is_valid(&self) -> bool {
        self.amp.is_finite()
            && self.amp > T::zero()
            && self.scale2.is_finite()
            && self.scale2 > T::zero()
            && self.center.is_finite()
            && self.eta.is_finite()
            && self.eta >= T::zero()
            && self.eta <= T::one()
    }

    fn is_significant(&self, precision: T) -> bool {
        self.maximum().abs() > precision && self.scale2.abs() > precision
    }
}

impl<T> Voigt<T>
where
    T: Float,
{
    /// Creates a new `Voigt` with the specified parameters.
    ///
    /// Note that `scale2` is the squared half-width at half-maximum, not the
    /// half-width itself, as outlined in the struct documentation.
    pub fn new(amp: T, scale2: T, center: T, eta: T) -> Self {
        Self {
            amp,
            scale2,
            center,
            eta,
        }
    }

    /// Creates a new `Voigt` from the specified parameters.
    pub fn from_untransformed(amp: T, scale: T, center: T, eta: T) -> Self {
        Self {
            amp,
            scale2: scale.powi(2),
            center,
            eta,
        }
    }
}
