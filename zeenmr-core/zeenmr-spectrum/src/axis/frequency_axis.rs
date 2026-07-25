use crate::axis::reference::ShiftReference;
use crate::range::{FiniteBounds, FrequencyRange, ShiftRange, SpectralRange};
use num_traits::Float;
use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
/// When the `serde` feature is enabled, `FrequencyAxis` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        try_from = "RawAxis<T>",
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
    larmor: T,
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
    pub fn new(range: FrequencyRange<T>, larmor: T, reference: ShiftReference<T>) -> Option<Self> {
        let offset = reference.offset(larmor)?;
        let start = offset + range.start() / larmor;
        let end = offset + range.end() / larmor;

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

    /// Attaches a length to the axis.
    pub fn grid(&self, len: usize) -> FrequencyGrid<'_, T> {
        FrequencyGrid { axis: self, len }
    }

    /// Returns the larmor frequency.
    pub fn larmor(&self) -> T {
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
            offset + self.range.start() / self.larmor,
            offset + self.range.end() / self.larmor,
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
        self.freq_step(len) / self.larmor
    }

    /// Converts a frequency to a chemical shift.
    ///
    /// Returns `None` if `freq` is not within the frequency range.
    pub fn freq_to_shift(&self, freq: T) -> Option<T> {
        if !self.range.contains(freq) {
            return None;
        }

        Some(self.reference.offset_unvalidated(self.larmor) + freq / self.larmor)
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
        let freq = (shift - offset) * self.larmor;

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

    /// Returns an iterator over `size` equally spaced frequencies spanning the
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
    pub fn freqs(&self, len: usize) -> AxisIter<T> {
        let step = self.freq_step(len);
        let start = self.range.start();

        AxisIter {
            start,
            step,
            front: 0,
            back: len,
        }
    }

    /// Returns an iterator over `size` equally spaced chemical shifts spanning
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
    pub fn shifts(&self, len: usize) -> AxisIter<T> {
        let step = self.shift_step(len);
        let start = self.shift_range().start();

        AxisIter {
            start,
            step,
            front: 0,
            back: len,
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> FrequencyAxis<T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over `size` equally spaced frequencies
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
    pub fn par_freqs(&self, len: usize) -> ParAxisIter<T> {
        let step = self.freq_step(len);
        let start = self.range.start();

        ParAxisIter(AxisIter {
            start,
            step,
            front: 0,
            back: len,
        })
    }

    /// Returns a parallel iterator over `size` equally spaced chemical shifts
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
    pub fn par_shifts(&self, len: usize) -> ParAxisIter<T> {
        let step = self.shift_step(len);
        let start = self.shift_range().start();

        ParAxisIter(AxisIter {
            start,
            step,
            front: 0,
            back: len,
        })
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
    pub fn freqs(&self) -> AxisIter<T> {
        self.axis.freqs(self.len)
    }

    /// Returns an iterator over the chemical shifts in the grid.
    ///
    /// See [`FrequencyAxis::shifts`] for its exact behavior.
    pub fn shifts(&self) -> AxisIter<T> {
        self.axis.shifts(self.len)
    }
}

#[cfg(feature = "rayon")]
impl<'axis, T> FrequencyGrid<'axis, T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over the frequencies in the grid.
    ///
    /// See [`FrequencyAxis::par_freqs`] for its exact behavior.
    pub fn par_freqs(&self) -> ParAxisIter<T> {
        self.axis.par_freqs(self.len)
    }

    /// Returns a parallel iterator over the chemical shifts in the grid.
    ///
    /// See [`FrequencyAxis::par_shifts`] for its exact behavior.
    pub fn par_shifts(&self) -> ParAxisIter<T> {
        self.axis.par_shifts(self.len)
    }
}

/// Iterator over evenly spaced elements spanning an axis.
#[derive(Clone, Debug)]
pub struct AxisIter<T> {
    /// Start of the range.
    start: T,
    /// Signed step size.
    step: T,
    /// Next element index from the front.
    front: usize,
    /// Next element index from the back + 1.
    back: usize,
}

impl<T> Iterator for AxisIter<T>
where
    T: Float,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = T::from(self.front).expect("conversion from usize to T must never fail");
            self.front += 1;

            Some(self.start + self.step * curr)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;

        (remaining, Some(remaining))
    }
}

impl<T> DoubleEndedIterator for AxisIter<T>
where
    T: Float,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            let curr = T::from(self.back).expect("conversion from usize to T must never fail");

            Some(self.start + self.step * curr)
        } else {
            None
        }
    }
}

impl<T> ExactSizeIterator for AxisIter<T> where T: Float {}

impl<T> FusedIterator for AxisIter<T> where T: Float {}

/// Parallel iterator over evenly spaced elements spanning an axis.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParAxisIter<T>(AxisIter<T>);

#[cfg(feature = "rayon")]
impl<T> ParallelIterator for ParAxisIter<T>
where
    T: Float + Send,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

#[cfg(feature = "rayon")]
impl<T> IndexedParallelIterator for ParAxisIter<T>
where
    T: Float + Send,
{
    fn len(&self) -> usize {
        self.0.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(AxisIterProducer(self.0))
    }
}

/// Producer for [`ParAxisIter`].
#[cfg(feature = "rayon")]
struct AxisIterProducer<T>(AxisIter<T>);

#[cfg(feature = "rayon")]
impl<T> Producer for AxisIterProducer<T>
where
    T: Float + Send,
{
    type Item = T;
    type IntoIter = AxisIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = AxisIter {
            back: mid,
            ..self.0
        };
        let right = AxisIter {
            front: mid,
            ..self.0
        };

        (Self(left), Self(right))
    }
}

/// Raw axis without invariants as an intermediate for deserialization.
#[cfg(feature = "serde")]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize),
    serde(bound(deserialize = "T: Float + Deserialize<'de>"))
)]
struct RawAxis<T> {
    /// Frequency range of the full axis.
    range: FrequencyRange<T>,
    /// Larmor frequency of the nucleus in the experiment.
    ///
    /// Stored in units of megahertz.
    larmor: T,
    /// Chemical shift reference.
    reference: ShiftReference<T>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<RawAxis<T>> for FrequencyAxis<T>
where
    T: Float,
{
    type Error = &'static str;

    fn try_from(value: RawAxis<T>) -> Result<Self, Self::Error> {
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

    fn test_parameters<T>() -> (FrequencyRange<T>, T, ShiftReference<T>)
    where
        T: Float,
    {
        let start = T::zero();
        let end = T::from(12000_u32).unwrap();
        let ref_freq = T::from(3000_u32).unwrap();
        let range = FrequencyRange::new(start, end).unwrap();
        let larmor = T::from(600.25_f64).unwrap();
        let reference = ShiftReference::from_freq(ref_freq).unwrap();

        (range, larmor, reference)
    }

    #[test]
    fn thread_safety() {
        assert_impl_all!(FrequencyAxis<f32>: Send, Sync);
        assert_impl_all!(FrequencyAxis<f64>: Send, Sync);
    }

    #[test]
    fn zero_larmor() {
        fn zero_larmor_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let larmor = T::zero();
            let (range, _, reference) = test_parameters();

            assert!(FrequencyAxis::new(range, larmor, reference).is_none());
        }

        zero_larmor_::<f32>();
        zero_larmor_::<f64>();
    }

    #[test]
    fn non_finite_larmor() {
        fn non_finite_larmor_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let larmors = [T::nan(), T::infinity(), T::neg_infinity()];
            let (range, _, reference) = test_parameters();

            for larmor in larmors {
                assert!(FrequencyAxis::new(range, larmor, reference).is_none());
            }
        }

        non_finite_larmor_::<f32>();
        non_finite_larmor_::<f64>();
    }

    #[test]
    fn incompatible_larmor() {
        fn incompatible_larmor_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let larmors = [T::min_positive_value(), -T::min_positive_value()];
            let (range, _, reference) = test_parameters();

            for larmor in larmors {
                assert!(FrequencyAxis::new(range, larmor, reference).is_none());
            }
        }

        incompatible_larmor_::<f32>();
        incompatible_larmor_::<f64>();
    }
}
