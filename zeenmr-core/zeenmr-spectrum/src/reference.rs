use num_traits::Zero;
use uom::si::f64::{Frequency, Ratio};

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
pub struct ShiftReference {
    /// Chemical shift of the reference.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Ratio::is_zero")
    )]
    shift: Ratio,
    /// Frequency that the chemical shift is anchored to.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Frequency::is_zero")
    )]
    frequency: Frequency,
}

impl ShiftReference {
    /// Maps the provided frequency to the provided chemical shift
    ///
    /// Returns `None` if either input is one of the infinities or `NaN`.
    pub fn new(shift: Ratio, frequency: Frequency) -> Option<Self> {
        if !(shift.is_finite() && frequency.is_finite()) {
            return None;
        }

        Some(Self { shift, frequency })
    }

    /// Maps zero on the frequency scale to the provided chemical shift.
    ///
    /// Returns `None` if `shift` is one of the infinities or `NaN`.
    pub fn from_shift(shift: Ratio) -> Option<Self> {
        Self::new(shift, Frequency::zero())
    }

    /// Maps the provided frequency to zero on the chemical shift scale.
    ///
    /// Returns `None` if `frequency` is one of the infinities or `NaN`.
    pub fn from_freq(frequency: Frequency) -> Option<Self> {
        Self::new(Ratio::zero(), frequency)
    }

    /// Returns the offset to apply to chemical shift values obtained by
    /// dividing frequencies by the larmor frequency.
    ///
    /// Returns `None` if `larmor` is zero, one of the infinities, or `NaN`.
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset(&self, larmor: Frequency) -> Option<Ratio> {
        if !larmor.is_finite() || larmor.is_zero() {
            return None;
        }

        Some(self.shift - self.frequency / larmor)
    }

    /// Returns the offset to apply to chemical shift values obtained by
    /// dividing frequencies by the larmor frequency, without validating
    /// `larmor`.
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset_unchecked(&self, larmor: Frequency) -> Ratio {
        self.shift - self.frequency / larmor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ShiftReference: Send, Sync);
    }
}
