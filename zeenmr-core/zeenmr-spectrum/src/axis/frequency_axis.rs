use crate::axis::iter::AxisValues;
use crate::axis::range::{FiniteBounds, FrequencyRange, ShiftRange, SpectralRange};
use num_traits::Float;
use std::cmp::Ordering;

#[cfg(feature = "rayon")]
use crate::axis::iter::ParAxisValues;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Larmor frequency of an NMR experiment.
///
/// The contained value is in units of megahertz.
///
/// # Invariants
///
/// The contained value must not be zero, `NaN`, or one of the infinities.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// With the `serde` feature enabled, `Larmor` can be serialized using `serde`.
/// The two directions have different requirements:
///
/// - [`Serialize`] needs only `T: Serialize`. A `Larmor` that exists is already
///   valid.
/// - [`Deserialize`] needs `T: Float + Deserialize<'de>`, so that the values
///   can be checked against the invariants.
///
/// [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
/// [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html
///
/// Deserialization goes through [`Larmor::new`] and fails if the value is
/// invalid according to the invariants.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        try_from = "RawLarmor<T>",
        bound(deserialize = "T: Float + Deserialize<'de>")
    )
)]
#[repr(transparent)]
pub struct Larmor<T>(T);

// we banished NaNs so this is okay as per Rust's Eq docs.
impl<T> Eq for Larmor<T> where T: Float {}

// we have to silence clippy because it can't understand that this only works
// for T: Float
#[allow(clippy::derive_ord_xor_partial_ord)]
// we banished NaNs so this is okay as per Rust's Ord docs.
impl<T> Ord for Larmor<T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("construction rejects NaN, so the ordering is total")
    }
}

impl<T> Larmor<T>
where
    T: Float,
{
    /// Creates a new larmor frequency.
    ///
    /// `freq` is in units of megahertz.
    ///
    /// Returns `None` if `freq` is zero, `NaN` or one of the infinities.
    pub fn new(freq: T) -> Option<Self> {
        if !freq.is_finite() || freq.is_zero() {
            return None;
        }

        Some(Self(freq))
    }

    /// Returns the contained value.
    pub fn get(&self) -> T {
        self.0
    }
}

#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct RawLarmor<T>(T);

#[cfg(feature = "serde")]
impl<T> TryFrom<RawLarmor<T>> for Larmor<T>
where
    T: Float,
{
    type Error = &'static str;

    fn try_from(value: RawLarmor<T>) -> Result<Self, Self::Error> {
        Self::new(value.0).ok_or("invalid larmor frequency value")
    }
}

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
    /// Returns `None` if the calculation below produces a non-finite offset.
    ///
    /// # Formulation
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset(&self, larmor: Larmor<T>) -> Option<T> {
        let offset = self.shift - self.freq / larmor.0;

        if offset.is_finite() {
            Some(offset)
        } else {
            None
        }
    }

    /// Returns the offset to apply to chemical shift values.
    ///
    /// Does not validate the resulting offset, which may be one of the
    /// infinities.
    ///
    /// # Formulation
    ///
    /// In order to calculate the chemical shift of a frequency value, divide it
    /// by the larmor frequency and add the offset to it:
    ///
    /// ```text
    /// shift = offset + frequency / larmor
    /// ```
    pub fn offset_unvalidated(&self, larmor: Larmor<T>) -> T {
        self.shift - self.freq / larmor.0
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

/// Frequency axis of a spectrum.
///
/// # Design
///
/// Readjusting the chemical shift reference is a common operation in NMR data
/// analysis, so we only store the information necessary to compute frequency
/// and chemical shift values on the fly, rather than caching them in memory.
///
/// # Invariants
///
/// When considering a [`ShiftReference`] and a larmor frequency, we say they
/// are compatible if [`ShiftReference::offset`] returns a `Some` value when
/// passed the larmor frequency. The invariant of `FrequencyAxis` is that its
/// chemical shift reference is compatible with its larmor frequency, and that
/// every frequency value within its range maps to a finite chemical shift
/// value. The latter always holds as long as it holds at the boundaries.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// With the `serde` feature enabled, `FrequencyAxis` can be serialized using
/// `serde`. The two directions have different requirements:
///
/// - [`Serialize`] needs only `T: Serialize`.
/// - [`Deserialize`] needs `T: Float + Deserialize<'de>` so that the values can
///   be checked against the invariants.
///
/// [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
/// [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html
///
/// Deserialization goes through [`FrequencyAxis::new`] and fails if the values
/// are invalid according to the invariants.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        try_from = "RawFrequencyAxis<T>",
        bound(deserialize = "T: Float + Deserialize<'de>")
    )
)]
pub struct FrequencyAxis<T> {
    /// Frequency range of the full axis.
    range: FrequencyRange<T>,
    /// Larmor frequency of the nucleus in the experiment.
    ///
    /// Stored in units of megahertz. Used to convert frequency values to
    /// chemical shifts. Also known as operating frequency and spectrometer
    /// frequency.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    larmor: Larmor<T>,
    /// Chemical shift reference.
    reference: ShiftReference<T>,
}

impl<T> FrequencyAxis<T>
where
    T: Float,
{
    /// Constructs a new `FrequencyAxis`.
    ///
    /// Returns `None` if `larmor` is incompatible with `reference`, or if the
    /// frequency range boundaries map to non-finite chemical shifts.
    pub fn new(
        range: FrequencyRange<T>,
        larmor: Larmor<T>,
        reference: ShiftReference<T>,
    ) -> Option<Self> {
        let offset = reference.offset(larmor)?;
        let start = offset + range.start() / larmor.0;
        let end = offset + range.end() / larmor.0;

        if !(start.is_finite() && end.is_finite()) {
            return None;
        }

        Some(Self {
            range,
            larmor,
            reference,
        })
    }

    /// Sets a new chemical shift reference.
    ///
    /// Returns `None` under the same conditions as [`FrequencyAxis::new`].
    pub fn with_reference(self, reference: ShiftReference<T>) -> Option<Self> {
        Self::new(self.range, self.larmor, reference)
    }

    /// Attaches a length to the axis, producing a grid.
    ///
    /// # Precision
    ///
    /// Passing a `len` that cannot be represented by `T` may lead to
    /// significant errors (e.g., `len > 2^24` for `f32`) in the downstream
    /// methods.
    pub fn grid(&self, len: usize) -> FrequencyGrid<'_, T> {
        FrequencyGrid { axis: self, len }
    }

    /// Returns the larmor frequency.
    pub fn larmor(&self) -> Larmor<T> {
        self.larmor
    }

    /// Returns the shift reference.
    pub fn reference(&self) -> ShiftReference<T> {
        self.reference
    }

    /// Returns the frequency range.
    pub fn freq_range(&self) -> FrequencyRange<T> {
        self.range
    }

    /// Returns the chemical shift range.
    pub fn shift_range(&self) -> ShiftRange<T> {
        let offset = self.reference.offset_unvalidated(self.larmor);

        // never panics because `new` would return `None` in the cases it could
        ShiftRange::new(
            offset + self.range.start() / self.larmor.0,
            offset + self.range.end() / self.larmor.0,
        )
        .expect("invariants guarantee that this is fine")
    }

    /// Returns the frequency step size given a number of data points.
    ///
    /// The step size is positive for ascending frequency ranges, negative for
    /// descending ones.
    ///
    /// If `len <= 1`, the returned step size is `0`.
    pub fn freq_step(&self, len: usize) -> T {
        if len <= 1 {
            T::zero()
        } else {
            self.range.signed_width()
                / T::from(len - 1).expect("conversion from usize to T must never fail")
        }
    }

    /// Returns the chemical shift step size given a number of data points.
    ///
    /// The step size is positive for ascending chemical shift ranges, negative
    /// for descending ones.
    ///
    /// If `len <= 1`, the returned step size is `0`.
    pub fn shift_step(&self, len: usize) -> T {
        self.freq_step(len) / self.larmor.0
    }

    /// Converts a frequency to a chemical shift.
    ///
    /// Returns `None` if `freq` is not within the frequency range.
    pub fn freq_to_shift(&self, freq: T) -> Option<T> {
        if !self.range.contains(freq) {
            return None;
        }

        Some(self.reference.offset_unvalidated(self.larmor) + freq / self.larmor.0)
    }

    /// Converts a chemical shift to a frequency.
    ///
    /// The returned frequency is clamped to the frequency range to avoid round
    /// trips landing slightly outside due to rounding.
    ///
    /// Returns `None` if `shift` is not within the chemical shift range.
    pub fn shift_to_freq(&self, shift: T) -> Option<T> {
        if !self.shift_range().contains(shift) {
            return None;
        }

        let offset = self.reference.offset_unvalidated(self.larmor);
        let freq = (shift - offset) * self.larmor.0;

        Some(freq.clamp(self.range.lower(), self.range.upper()))
    }

    /// Converts a frequency to a relative coordinate in terms of the width.
    ///
    /// Returns `None` if `freq` is not within the frequency range, or if said
    /// range has a width of `0`.
    pub fn freq_to_rel(&self, freq: T) -> Option<T> {
        if !self.range.contains(freq) || self.range.width().is_zero() {
            return None;
        }

        Some((freq - self.range.start()) / self.range.signed_width())
    }

    /// Converts a chemical shift to a relative coordinate in terms of the
    /// width.
    ///
    /// Returns `None` if `shift` is not within the chemical shift range, or if
    /// said range has a width of `0`.
    pub fn shift_to_rel(&self, shift: T) -> Option<T> {
        let shift_range = self.shift_range();

        if !shift_range.contains(shift) || shift_range.width().is_zero() {
            return None;
        }

        Some((shift - shift_range.start()) / shift_range.signed_width())
    }

    /// Converts a relative coordinate in terms of the width to a frequency.
    ///
    /// The returned frequency is clamped to the frequency range to avoid round
    /// trips landing slightly outside due to rounding.
    ///
    /// Returns `None` if `rel ∉ [0, 1]`, or if the frequency range has a width
    /// of `0`.
    pub fn rel_to_freq(&self, rel: T) -> Option<T> {
        if !(T::zero()..=T::one()).contains(&rel) || self.range.width().is_zero() {
            return None;
        }

        let freq = self.range.start() + rel * self.range.signed_width();

        Some(freq.clamp(self.range.lower(), self.range.upper()))
    }

    /// Converts a relative coordinate in terms of the width to a chemical
    /// shift.
    ///
    /// The returned chemical shift is clamped to the chemical shift range to
    /// avoid round trips landing slightly outside due to rounding.
    ///
    /// Returns `None` if `rel ∉ [0, 1]`, or if the chemical shift range has a
    /// width of `0`.
    pub fn rel_to_shift(&self, rel: T) -> Option<T> {
        let shift_range = self.shift_range();

        if !(T::zero()..=T::one()).contains(&rel) || shift_range.width().is_zero() {
            return None;
        }

        let shift = shift_range.start() + rel * shift_range.signed_width();

        Some(shift.clamp(shift_range.lower(), shift_range.upper()))
    }

    /// Returns an iterator over `len` equally spaced frequencies spanning the
    /// axis.
    ///
    /// Each call of this method recomputes the frequencies on the fly.
    ///
    /// # Length
    ///
    /// If `len` is `0`, the returned iterator is simply empty. If `len` is
    /// `1`, the returned iterator contains only the start value of the range.
    /// If the width of the range is `0`, the returned iterator repeats the
    /// start value of the range `len` times.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// may not be exactly identical to the end of the frequency range. Passing
    /// a `len` that cannot be represented by `T` may also lead to significant
    /// errors (e.g., `len > 2^24` for `f32`).
    pub fn freqs(&self, len: usize) -> AxisValues<T> {
        let start = self.range.start();
        let step = self.freq_step(len);

        AxisValues::new(start, step, len)
    }

    /// Returns an iterator over `len` equally spaced chemical shifts spanning
    /// the axis.
    ///
    /// Each call of this method recomputes the chemical shifts on the fly.
    ///
    /// # Length
    ///
    /// If `len` is `0`, the returned iterator is simply empty. If `len` is
    /// `1`, the returned iterator contains only the start value of the range.
    /// If the width of the range is `0`, the returned iterator repeats the
    /// start value of the range `len` times.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// may not be exactly identical to the end of the frequency range. Passing
    /// a `len` that cannot be represented by `T` may also lead to significant
    /// errors (e.g., `len > 2^24` for `f32`).
    pub fn shifts(&self, len: usize) -> AxisValues<T> {
        let start = self.shift_range().start();
        let step = self.shift_step(len);

        AxisValues::new(start, step, len)
    }
}

#[cfg(feature = "rayon")]
impl<T> FrequencyAxis<T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over `len` equally spaced frequencies
    /// spanning the axis.
    ///
    /// Each call of this method recomputes the frequencies on the fly.
    ///
    /// # Length
    ///
    /// If `len` is `0`, the returned iterator is simply empty. If `len` is
    /// `1`, the returned iterator contains only the start value of the range.
    /// If the width of the range is `0`, the returned iterator repeats the
    /// start value of the range `len` times.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// may not be exactly identical to the end of the frequency range. Passing
    /// a `len` that cannot be represented by `T` may also lead to significant
    /// errors (e.g., `len > 2^24` for `f32`).
    pub fn par_freqs(&self, len: usize) -> ParAxisValues<T> {
        let start = self.range.start();
        let step = self.freq_step(len);

        ParAxisValues::new(AxisValues::new(start, step, len))
    }

    /// Returns a parallel iterator over `len` equally spaced chemical shifts
    /// spanning the axis.
    ///
    /// Each call of this method recomputes the chemical shifts on the fly.
    ///
    /// # Length
    ///
    /// If `len` is `0`, the returned iterator is simply empty. If `len` is
    /// `1`, the returned iterator contains only the start value of the range.
    /// If the width of the range is `0`, the returned iterator repeats the
    /// start value of the range `len` times.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// may not be exactly identical to the end of the frequency range. Passing
    /// a `len` that cannot be represented by `T` may also lead to significant
    /// errors (e.g., `len > 2^24` for `f32`).
    pub fn par_shifts(&self, len: usize) -> ParAxisValues<T> {
        let start = self.shift_range().start();
        let step = self.shift_step(len);

        ParAxisValues::new(AxisValues::new(start, step, len))
    }
}

/// Convenience wrapper of an axis reference with an attached length.
///
/// An instance of this type can be obtained from [`FrequencyAxis::grid`].
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct FrequencyGrid<'axis, T> {
    /// Reference to the original axis.
    ///
    /// This is not owned in order to enforce single source of truth.
    axis: &'axis FrequencyAxis<T>,
    /// Length of the 1D grid.
    len: usize,
}

impl<'axis, T> FrequencyGrid<'axis, T>
where
    T: Float,
{
    /// Sets the length of the grid.
    pub fn with_len(self, len: usize) -> Self {
        Self { len, ..self }
    }

    /// Returns a reference to the original axis.
    pub fn axis(&self) -> &'axis FrequencyAxis<T> {
        self.axis
    }

    /// Returns the length of the grid.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the grid contains no points.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the frequency step size of the grid.
    ///
    /// See [`FrequencyAxis::freq_step`] for its exact behavior.
    pub fn freq_step(&self) -> T {
        self.axis.freq_step(self.len)
    }

    /// Returns the chemical shift step size of the grid.
    ///
    /// See [`FrequencyAxis::shift_step`] for its exact behavior.
    pub fn shift_step(&self) -> T {
        self.axis.shift_step(self.len)
    }

    /// Returns an iterator over the frequencies in the grid.
    ///
    /// See [`FrequencyAxis::freqs`] for its exact behavior.
    pub fn freqs(&self) -> AxisValues<T> {
        self.axis.freqs(self.len)
    }

    /// Returns an iterator over the chemical shifts in the grid.
    ///
    /// See [`FrequencyAxis::shifts`] for its exact behavior.
    pub fn shifts(&self) -> AxisValues<T> {
        self.axis.shifts(self.len)
    }
}

#[cfg(feature = "rayon")]
impl<T> FrequencyGrid<'_, T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over the frequencies in the grid.
    ///
    /// See [`FrequencyAxis::par_freqs`] for its exact behavior.
    pub fn par_freqs(&self) -> ParAxisValues<T> {
        self.axis.par_freqs(self.len)
    }

    /// Returns a parallel iterator over the chemical shifts in the grid.
    ///
    /// See [`FrequencyAxis::par_shifts`] for its exact behavior.
    pub fn par_shifts(&self) -> ParAxisValues<T> {
        self.axis.par_shifts(self.len)
    }
}

/// Raw axis without invariants as an intermediate for deserialization.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(bound(deserialize = "T: Float + Deserialize<'de>"))]
struct RawFrequencyAxis<T> {
    /// Frequency range of the full axis.
    range: FrequencyRange<T>,
    /// Larmor frequency of the nucleus in the experiment.
    ///
    /// Stored in units of megahertz.
    larmor: Larmor<T>,
    /// Chemical shift reference.
    reference: ShiftReference<T>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<RawFrequencyAxis<T>> for FrequencyAxis<T>
where
    T: Float,
{
    type Error = &'static str;

    fn try_from(value: RawFrequencyAxis<T>) -> Result<Self, Self::Error> {
        Self::new(value.range, value.larmor, value.reference).ok_or(
            "incompatible chemical shift reference and larmor frequency, \
                 or non-finite computed chemical shift bounds",
        )
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
        assert_impl_all!(FrequencyAxis<f32>: Send, Sync);
        assert_impl_all!(FrequencyAxis<f64>: Send, Sync);
    }
}
