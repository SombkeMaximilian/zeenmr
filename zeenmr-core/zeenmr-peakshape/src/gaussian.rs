use crate::approximations::Exp2;
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
/// type parameter is [`Exp2`] + [`FloatConst`].
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
/// Two transformations are applied to the exponent. First, the reciprocal of
/// the doubled, squared scale is absorbed into a single factor, removing a
/// division and two multiplications from every evaluation. Second, the base is
/// changed from `e` to `2`, since `exp2` is the primitive that maps onto the
/// hardware: the integer part of a base-2 exponent *is* the exponent field of
/// an IEEE-754 float, so it can be constructed with a shift rather than
/// computed. The base conversion factor `log₂(e)` is folded into the same term,
/// so it never appears in the inner loop. Together these give `exp2_scale`:
///
/// ```text
/// exp2_scale = -log₂(e) / (2 c²)
/// ```
///
/// And the finalized transformed function:
///
/// ```text
/// f(x) = amp * 2^((x - center)² * exp2_scale)
/// ```
///
/// Note that `exp2_scale` is negative; the sign of the exponent is absorbed
/// along with everything else.
///
/// This representation is used internally for efficiency, but the original
/// parameter can be recovered by inverting the transformation:
///
/// ```text
/// scale = sqrt(-log₂(e) / (2 * exp2_scale))
/// ```
///
/// # Out-of-Range Parameters
///
/// For a `Gaussian` to represent a peak shape, `amp` must be positive and
/// `exp2_scale` must be negative. This is not enforced at construction to
/// avoid unnecessary overhead.
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
    /// The height of the peak, i.e. `f(center) = amp`. Must be positive.
    amp: T,
    /// Base-2 exponent coefficient, `-log₂(e) / (2 c²)`.
    ///
    /// Absorbs the negation, the doubled squared scale, and the conversion from
    /// base `e` to base `2`. Must be negative.
    exp2_scale: T,
    /// Center of the Gaussian function.
    center: T,
}

impl<T> FromArray<T, 3> for Gaussian<T>
where
    T: Exp2,
{
    /// Creates a new `Gaussian` from its parameter array.
    ///
    /// Order: `amp`, `exp2_scale`, `center`.
    fn from_array(array: [T; 3]) -> Self {
        Self::new(array[0], -array[1], array[2])
    }
}

impl<T> Evaluate<T> for Gaussian<T>
where
    T: Exp2,
{
    fn evaluate(&self, at: T) -> T {
        self.amp * ((at - self.center).powi(2) * self.exp2_scale).exp2_fast_nonpos()
    }
}

impl<T> PeakShape<T> for Gaussian<T>
where
    T: Exp2 + FloatConst,
{
    fn center(&self) -> T {
        self.center
    }

    fn maximum(&self) -> T {
        self.amp
    }

    fn half_width(&self) -> T {
        (-self.exp2_scale.recip()).sqrt()
    }

    fn full_width(&self) -> T {
        (T::one() + T::one()) * self.half_width()
    }

    fn area(&self) -> T {
        self.amp * (-T::PI() / (self.exp2_scale * T::LN_2())).sqrt()
    }

    fn is_valid(&self) -> bool {
        self.amp.is_finite()
            && self.amp > T::zero()
            && self.exp2_scale.is_finite()
            && self.exp2_scale < T::zero()
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
    pub fn new(amp: T, exp2_scale: T, center: T) -> Self {
        Self {
            amp,
            exp2_scale,
            center,
        }
    }
}

impl<T> Gaussian<T>
where
    T: Float + FloatConst,
{
    /// Creates a new `Gaussian` from the untransformed parameters.
    pub fn from_untransformed(amp: T, scale: T, center: T) -> Self {
        Self {
            amp,
            exp2_scale: -T::LOG2_E() / (T::one() + T::one() * scale.powi(2)),
            center,
        }
    }
}
