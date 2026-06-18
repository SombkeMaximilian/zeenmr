use num_traits::Zero;
use uom::si::f64::Ratio;

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
#[derive(Clone, PartialEq, Debug, Default)]
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
    pub shift: Ratio,
    /// Index within the Spectrum that corresponds to the reference position.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "usize::is_zero")
    )]
    pub index: usize,
}

impl ShiftReference {
    /// Constructs a new `ShiftReference`.
    pub fn new(shift: Ratio, index: usize) -> Self {
        Self { shift, index }
    }

    /// Constructs a new `ShiftReference`.
    ///
    /// The index is set to `0`.
    pub fn from_shift(shift: Ratio) -> Self {
        Self::new(shift, 0)
    }

    /// Constructs a new `ShiftReference`.
    ///
    /// The shift is set to `0.0`.
    pub fn from_index(index: usize) -> Self {
        Self::new(Ratio::zero(), index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[cfg(feature = "serde")]
    use uom::si::ratio::part_per_million as ppm;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ShiftReference: Send, Sync);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let references = [
            ShiftReference::new(Ratio::new::<ppm>(14.0), 0),
            ShiftReference::new(Ratio::new::<ppm>(4.8), 2_usize.pow(13)),
            ShiftReference::new(Ratio::zero(), 12000),
        ];
        let serialized = references
            .clone()
            .map(|reference| serde_json::to_string(&reference).unwrap());
        let deserialized = serialized
            .clone()
            .map(|serialized| serde_json::from_str::<ShiftReference>(&serialized).unwrap());

        references
            .into_iter()
            .zip(deserialized)
            .for_each(|(init, rec)| assert_eq!(init, rec));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_missing_fields() {
        let serialized = [
            "{\"shift\": 0.0000048, \"index\": 16384}",
            "{\"shift\": 0.000014}",
            "{\"index\": 16384}",
        ];
        let expected = [
            ShiftReference::new(Ratio::new::<ppm>(4.8), 2_usize.pow(14)),
            ShiftReference::new(Ratio::new::<ppm>(14.0), 0),
            ShiftReference::new(Ratio::new::<ppm>(0.0), 2_usize.pow(14)),
        ];
        let deserialized =
            serialized.map(|reference| serde_json::from_str::<ShiftReference>(reference).unwrap());

        expected
            .into_iter()
            .zip(deserialized)
            .for_each(|(init, rec)| assert_eq!(init, rec));
    }
}
