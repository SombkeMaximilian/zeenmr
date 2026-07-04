use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a reference used for calibrating chemical shifts in the NMR
/// spectrum.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// If the `serde` feature is enabled, `ShiftReference` implements
/// [`Serialize`] and [`Deserialize`].
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Copy, Clone, PartialEq, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct ShiftReference<T> {
    /// Chemical shift of the reference.
    shift: T,
    /// Frequency that the chemical shift is anchored to.
    frequency: T,
}

impl<T> ShiftReference<T>
where
    T: Float,
{
    /// Creates a new `ShiftReference`.
    ///
    /// Returns `None` if either input is one of the infinities or `NaN`.
    ///
    /// Maps the provided frequency to the provided chemical shift
    pub fn new(shift: T, frequency: T) -> Option<Self> {
        if !(shift.is_finite() && frequency.is_finite()) {
            return None;
        }

        Some(Self { shift, frequency })
    }

    /// Creates a new `ShiftReference`.
    ///
    /// Returns `None` if `shift` is one of the infinities or `NaN`.
    ///
    /// Maps zero on the frequency scale to the provided chemical shift.
    pub fn from_shift(shift: T) -> Option<Self> {
        Self::new(shift, T::zero())
    }

    /// Creates a new `ShiftReference`.
    ///
    /// Returns `None` if `frequency` is one of the infinities or `NaN`.
    ///
    /// Maps the provided frequency to zero on the chemical shift scale.
    pub fn from_freq(frequency: T) -> Option<Self> {
        Self::new(T::zero(), frequency)
    }

    /// Returns the offset to apply to chemical shift values obtained by
    /// dividing frequencies by the larmor frequency.
    ///
    /// Returns `None` if `larmor` is zero, one of the infinities, or `NaN`.
    ///
    /// `larmor` is in units of megahertz.
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset(&self, larmor: T) -> Option<T> {
        if !larmor.is_finite() || larmor.is_zero() {
            return None;
        }

        Some(self.shift - self.frequency / larmor)
    }

    /// Returns the offset to apply to chemical shift values obtained by
    /// dividing frequencies by the larmor frequency, without validating
    /// `larmor`.
    ///
    /// `larmor` is in units of megahertz.
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset_unchecked(&self, larmor: T) -> T {
        self.shift - self.frequency / larmor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ShiftReference<f32>: Send, Sync);
        assert_impl_all!(ShiftReference<f64>: Send, Sync);
    }
}
