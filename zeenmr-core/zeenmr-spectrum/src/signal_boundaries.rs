#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Boundaries of the signal region in a [`Spectrum`].
///
/// Signal boundaries can be defined in three different ways:
/// - Relative: as a fraction of the total spectral range (between 0.0 and 1.0).
/// - Frequencies: as absolute frequency values in Hz.
/// - Chemical shifts: as absolute chemical shift values in ppm.
///
/// # Serialization with [Serde]
///
/// [Serde]: (https://serde.rs/)
///
/// When the `serde` feature is enabled, `SignalBoundaries` implements the
/// [`Serialize`] and [`Deserialize`] traits.
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub enum SignalBoundaries {
    /// Relative boundaries defined as a fraction of the total spectral range.
    ///
    /// Values must be between 0.0 and 1.0, with 0.0 representing the left edge
    /// and 1.0 representing the right edge of the spectrum.
    Relative(f64, f64),
    /// Frequency boundaries in Hz.
    Frequencies(f64, f64),
    /// Chemical shift boundaries in ppm.
    ///
    /// These values require context to be interpreted. Providing boundaries in
    ChemicalShifts(f64, f64),
}

impl Default for SignalBoundaries {
    fn default() -> Self {
        SignalBoundaries::Relative(0.1, 0.9)
    }
}

impl From<SignalBoundaries> for (f64, f64) {
    fn from(value: SignalBoundaries) -> Self {
        match value {
            SignalBoundaries::Relative(start, end) => (start, end),
            SignalBoundaries::Frequencies(start, end) => (start, end),
            SignalBoundaries::ChemicalShifts(start, end) => (start, end),
        }
    }
}

impl SignalBoundaries {
    /// Returns the left boundary.
    pub fn start(&self) -> f64 {
        match self {
            SignalBoundaries::Relative(start, _) => *start,
            SignalBoundaries::Frequencies(start, _) => *start,
            SignalBoundaries::ChemicalShifts(start, _) => *start,
        }
    }

    /// Returns the right boundary.
    pub fn end(&self) -> f64 {
        match self {
            SignalBoundaries::Relative(_, end) => *end,
            SignalBoundaries::Frequencies(_, end) => *end,
            SignalBoundaries::ChemicalShifts(_, end) => *end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(SignalBoundaries: Send, Sync);
    }
}
