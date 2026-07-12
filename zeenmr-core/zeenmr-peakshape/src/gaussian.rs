use crate::{Evaluate, FromArray, PeakShape};
use num_traits::{Float, FloatConst};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a [Gaussian] peak shape.
///
/// [Gaussian]: https://en.wikipedia.org/wiki/Gaussian_function
///
/// Also known as the normal distribution.
///
/// This type implements the [`Evaluate`] trait and is a [`PeakShape`], if its
/// type parameter is [`Float`] + [`FloatConst`].
///
/// # Definition
///
/// The [Gaussian] function is typically defined as:
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
/// multiplications during every evaluation. The [Gaussian] function then
/// becomes:
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
/// # Negative Parameters
///
/// For a `Gaussian` to represent a peak shape, `amp` and `double_scale2` must
/// be positive. This is not enforced at construction to avoid unnecessary
/// overhead.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Gaussian` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gaussian<T> {
    /// Amplitude of the Gaussian function.
    ///
    /// Must be positive.
    amp: T,
    /// Doubled square of the standard deviation of the Gaussian function.
    ///
    /// Absorbed doubled, squared scale in the denominator of the exponential.
    /// Must be positive.
    double_scale2: T,
    /// Center of the Gaussian function.
    center: T,
}

impl<T> FromArray<T, 3> for Gaussian<T>
where
    T: Float,
{
    /// Creates a new `Gaussian` from its parameter array.
    ///
    /// Order: `amp`, `double_scale2`, `center`.
    fn from_array(array: [T; 3]) -> Self {
        Self::new(array[0], array[1], array[2])
    }
}

impl<T> Evaluate<T> for Gaussian<T>
where
    T: Float,
{
    fn evaluate(&self, at: T) -> T {
        self.amp * (-(at - self.center).powi(2) / self.double_scale2).exp()
    }
}

impl<T> PeakShape<T> for Gaussian<T>
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
        (T::LN_2() * self.double_scale2).sqrt()
    }

    fn full_width(&self) -> T {
        (T::one() + T::one()) * self.half_width()
    }

    fn area(&self) -> T {
        self.amp * (T::PI() * self.double_scale2).sqrt()
    }

    fn is_valid(&self) -> bool {
        self.amp.is_finite()
            && self.amp > T::zero()
            && self.double_scale2.is_finite()
            && self.double_scale2 > T::zero()
            && self.center.is_finite()
    }

    fn is_significant(&self, precision: T) -> bool {
        self.maximum().abs() > precision && self.half_width().abs() > precision
    }
}

impl<T> Gaussian<T>
where
    T: Float,
{
    /// Creates a new `Gaussian` with the specified parameters.
    ///
    /// Note that these are not the standard parameters, but the transformed
    /// parameters as outlined in the struct documentation.
    pub fn new(amp: T, double_scale2: T, center: T) -> Self {
        Self {
            amp,
            double_scale2,
            center,
        }
    }

    /// Creates a new `Gaussian` from the untransformed parameters.
    pub fn from_untransformed(amp: T, scale: T, center: T) -> Self {
        Self {
            amp,
            double_scale2: (T::one() + T::one()) * scale.powi(2),
            center,
        }
    }
}
