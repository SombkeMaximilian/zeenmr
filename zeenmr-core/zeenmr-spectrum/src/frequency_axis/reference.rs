use num_traits::Float;
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a reference used for calibrating chemical shifts in the NMR
/// spectrum.
///
/// # Invariants
///
/// Neither value may be `NaN` nor one of the infinities, and the frequency must
/// not be negative.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// With the `serde` feature enabled, `ShiftReference` can be serialized using
/// `serde`. The two directions have different requirements:
///
/// - [`Serialize`] needs only `T: Serialize`. A `ShiftReference` that exists is
///   already valid.
/// - [`Deserialize`] needs `T: Float + Deserialize<'de>`, so that the values
///   can be checked against the invariants.
///
/// [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
/// [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html
///
/// Deserialization goes through [`ShiftReference::new`] and fails if the values
/// are invalid according to the invariants.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        try_from = "RawShiftReference<T>",
        bound(deserialize = "T: Float + Deserialize<'de>")
    )
)]
pub struct ShiftReference<T> {
    /// Chemical shift of the reference.
    shift: T,
    /// Frequency that the chemical shift is anchored to.
    freq: T,
}

impl<T> Default for ShiftReference<T>
where
    T: Float,
{
    /// Returns the identity reference: 0 ppm anchored at 0 Hz.
    fn default() -> Self {
        Self {
            shift: T::zero(),
            freq: T::zero(),
        }
    }
}

// we banished NaNs so this is okay as per Rust's Eq docs.
impl<T> Eq for ShiftReference<T> where T: Float {}

// we have to silence clippy because it can't understand that this only works
// for T: Float
#[allow(clippy::derive_ord_xor_partial_ord)]
// we banished NaNs so this is okay as per Rust's Ord docs.
impl<T> Ord for ShiftReference<T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("construction rejects NaN, so the ordering is total")
    }
}

impl<T> ShiftReference<T>
where
    T: Float,
{
    /// Creates a new `ShiftReference`.
    ///
    /// Maps the provided frequency to the provided chemical shift.
    ///
    /// Returns `None` if either input is `NaN` or one of the infinities, or if
    /// `frequency` is negative.
    pub fn new(shift: T, freq: T) -> Option<Self> {
        if !(shift.is_finite() && freq.is_finite() && freq >= T::zero()) {
            return None;
        }

        Some(Self { shift, freq })
    }

    /// Creates a new `ShiftReference`.
    ///
    /// Maps zero on the frequency scale to the provided chemical shift.
    ///
    /// Returns `None` if `shift` is one of the infinities or `NaN`.
    pub fn from_shift(shift: T) -> Option<Self> {
        Self::new(shift, T::zero())
    }

    /// Creates a new `ShiftReference`.
    ///
    /// Maps the provided frequency to zero on the chemical shift scale.
    ///
    /// Returns `None` if `frequency` is negative, `NaN`, or one of the
    /// infinities.
    pub fn from_freq(freq: T) -> Option<Self> {
        Self::new(T::zero(), freq)
    }

    /// Returns the offset to apply to chemical shift values.
    ///
    /// Returns `None` if `larmor` is zero, one of the infinities, or `NaN`, or
    /// if the calculation below produces a non-finite offset.
    ///
    /// `larmor` is in units of megahertz.
    ///
    /// # Formulation
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
        let offset = self.shift - self.freq / larmor;

        if offset.is_finite() {
            Some(offset)
        } else {
            None
        }
    }

    /// Returns the offset to apply to chemical shift values.
    ///
    /// Does not validate `larmor` or the resulting offset.
    ///
    /// `larmor` is in units of megahertz.
    ///
    /// # Formulation
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset_unvalidated(&self, larmor: T) -> T {
        self.shift - self.freq / larmor
    }
}

/// Raw chemical shift reference without invariants as an intermediate for
/// deserialization.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct RawShiftReference<T> {
    /// Chemical shift of the reference.
    shift: T,
    /// Frequency that the chemical shift is anchored to.
    freq: T,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<RawShiftReference<T>> for ShiftReference<T>
where
    T: Float,
{
    type Error = &'static str;

    fn try_from(value: RawShiftReference<T>) -> Result<Self, Self::Error> {
        Self::new(value.shift, value.freq).ok_or("invalid chemical shift reference values")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::{ApproxEq, assert_approx_eq};
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ShiftReference<f32>: Send, Sync);
        assert_impl_all!(ShiftReference<f64>: Send, Sync);
    }

    #[test]
    fn invariants() {
        fn invariants_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let zero = T::zero();
            let one = T::one();
            let nan = T::nan();
            let inf = T::infinity();
            let neg_inf = T::neg_infinity();

            assert!(ShiftReference::new(zero, -one).is_none());
            assert!(ShiftReference::new(zero, nan).is_none());
            assert!(ShiftReference::new(zero, inf).is_none());
            assert!(ShiftReference::new(zero, neg_inf).is_none());
            assert!(ShiftReference::new(nan, zero).is_none());
            assert!(ShiftReference::new(inf, zero).is_none());
            assert!(ShiftReference::new(neg_inf, zero).is_none());
            assert!(ShiftReference::new(nan, nan).is_none());
            assert!(ShiftReference::new(inf, inf).is_none());
            assert!(ShiftReference::new(neg_inf, neg_inf).is_none());
        }

        invariants_::<f32>();
        invariants_::<f64>();
    }

    #[test]
    fn offset() {
        fn offset_<T>()
        where
            T: Float + ApproxEq + std::fmt::Debug,
        {
            let zero = T::zero();
            let nan = T::nan();
            let inf = T::infinity();
            let neg_inf = T::neg_infinity();
            let min_pos = T::min_positive_value();
            let shift_refs = [
                ShiftReference::from_shift(T::from(-5).unwrap()),
                ShiftReference::from_freq(T::from(3000).unwrap()),
                ShiftReference::new(T::from(10).unwrap(), T::from(7500).unwrap()),
            ]
            .map(Option::unwrap);

            for shift_ref in shift_refs.iter() {
                assert!(shift_ref.offset(zero).is_none());
                assert!(shift_ref.offset(nan).is_none());
                assert!(shift_ref.offset(inf).is_none());
                assert!(shift_ref.offset(neg_inf).is_none());
                if !shift_ref.freq.is_zero() {
                    assert!(shift_ref.offset(min_pos).is_none());
                }
            }

            let larmor = T::from(625).unwrap();
            let offsets = [
                T::from(-5).unwrap(),
                T::from(-4.8).unwrap(),
                T::from(-2).unwrap(),
            ];

            for (shift_ref, offset) in shift_refs.into_iter().zip(offsets) {
                assert_approx_eq!(T, shift_ref.offset(larmor).unwrap(), offset);
            }
        }

        offset_::<f32>();
        offset_::<f64>();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        fn round_trip_<T>()
        where
            T: Float + Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
        {
            let shift_refs = [
                ShiftReference::from_shift(T::from(-5).unwrap()),
                ShiftReference::from_freq(T::from(3000).unwrap()),
                ShiftReference::new(T::from(10).unwrap(), T::from(7500).unwrap()),
            ]
            .map(Option::unwrap);

            for shift_ref in shift_refs {
                let ser = serde_json5::to_string(&shift_ref).unwrap();
                let de = serde_json5::from_str::<ShiftReference<T>>(&ser).unwrap();

                assert_eq!(shift_ref, de);
            }
        }

        round_trip_::<f32>();
        round_trip_::<f64>();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_invariants() {
        fn deserialization_invariants_<T>()
        where
            T: Float + serde::de::DeserializeOwned + std::fmt::Debug,
        {
            let bad_shift_refs = [
                "{ \"shift\": 0.0, \"freq\": -1.0 }",
                "{ \"shift\": 0.0, \"freq\": NaN }",
                "{ \"shift\": 0.0, \"freq\": Infinity }",
                "{ \"shift\": 0.0, \"freq\": -Infinity }",
                "{ \"shift\": NaN, \"freq\": 0.0 }",
                "{ \"shift\": Infinity, \"freq\": 0.0 }",
                "{ \"shift\": -Infinity, \"freq\": 0.0 }",
                "{ \"shift\": NaN, \"freq\": -1.0 }",
                "{ \"shift\": NaN, \"freq\": NaN }",
                "{ \"shift\": Infinity, \"freq\": Infinity }",
                "{ \"shift\": -Infinity, \"freq\": -Infinity }",
            ];

            for bad_shift_ref in bad_shift_refs {
                let err = serde_json5::from_str::<ShiftReference<T>>(bad_shift_ref)
                    .map_err(|e| e.to_string())
                    .unwrap_err();

                assert!(err.contains("invalid chemical shift reference values"));
            }
        }

        deserialization_invariants_::<f32>();
        deserialization_invariants_::<f64>();
    }
}
