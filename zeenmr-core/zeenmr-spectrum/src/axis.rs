use crate::range::{FrequencyRange, ShiftRange};
use crate::reference::ShiftReference;
use num_traits::Zero;
use uom::si::f64::{Frequency, Ratio};
use uom::si::ratio::ratio;

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
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Axis` can be serialized and
/// deserialized using `serde`.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct Axis {
    /// Frequency range of the full axis.
    range: FrequencyRange,
    /// Larmor frequency of the nucleus used to convert frequency values to
    /// chemical shifts.
    ///
    /// Typical order of magnitude is 1e6 Hz. Also known as operating frequency
    /// and spectrometer frequency.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    larmor: Frequency,
    /// Chemical shift reference.
    reference: ShiftReference,
}

impl Axis {
    /// Constructs a new `Axis`
    ///
    /// Returns `None` if the width of `range` is zero, or if `larmor` is one
    /// of the infinities, zero or negative.
    pub fn new(
        range: FrequencyRange,
        larmor: Frequency,
        reference: ShiftReference,
    ) -> Option<Self> {
        if range.width() == Frequency::zero() || !larmor.is_finite() || larmor <= Frequency::zero()
        {
            return None;
        }

        Some(Self {
            range,
            larmor,
            reference,
        })
    }

    /// Returns the larmor frequency.
    pub fn larmor(&self) -> Frequency {
        self.larmor
    }

    /// Returns the shift reference.
    pub fn reference(&self) -> ShiftReference {
        self.reference
    }

    /// Returns the frequency range.
    pub fn freq_range(&self) -> FrequencyRange {
        self.range
    }

    /// Returns the chemical shift range.
    pub fn shift_range(&self) -> ShiftRange {
        let offset = self.reference.offset_unchecked(self.larmor);

        ShiftRange::new(
            self.range.start() / self.larmor + offset,
            self.range.end() / self.larmor + offset,
        )
        .expect("range, larmor and reference should be validated at construction")
    }

    /// Returns the frequency step size given a number of data points.
    ///
    /// If `size <= 1`, the returned step size is not meaningful.
    pub fn freq_step(&self, size: usize) -> Frequency {
        self.range.signed_width() / (size.saturating_sub(1) as f64)
    }

    /// Returns the chemical shift step size given a number of data points.
    ///
    /// If `size <= 1`, the returned step size is not meaningful.
    pub fn shift_step(&self, size: usize) -> Ratio {
        self.freq_step(size) / self.larmor
    }

    /// Converts a frequency to a chemical shift.
    pub fn freq_to_shift(&self, freq: Frequency) -> Option<Ratio> {
        if !self.range.contains(freq) {
            return None;
        }

        Some(freq / self.larmor + self.reference.offset_unchecked(self.larmor))
    }

    /// Converts a chemical shift to a frequency.
    pub fn shift_to_freq(&self, shift: Ratio) -> Option<Frequency> {
        if !self.shift_range().contains(shift) {
            return None;
        }

        Some((shift - self.reference.offset_unchecked(self.larmor)) * self.larmor)
    }

    /// Converts a frequency to a relative coordinate in terms of the width.
    ///
    /// Returns `None` if `freq` is not within the frequency range.
    pub fn freq_to_rel(&self, freq: Frequency) -> Option<f64> {
        if !self.range.contains(freq) {
            return None;
        }

        Some(((freq - self.range.start()) / self.range.signed_width()).get::<ratio>())
    }

    /// Converts a chemical shift to a relative coordinate in terms of the
    /// width.
    ///
    /// Returns `None` if `shift` is not within the chemical shift range.
    pub fn shift_to_rel(&self, shift: Ratio) -> Option<f64> {
        let shift_range = self.shift_range();
        if !shift_range.contains(shift) {
            return None;
        }

        Some(((shift - shift_range.start()) / shift_range.signed_width()).get::<ratio>())
    }

    /// Converts a relative coordinate in terms of the total width to a
    /// frequency.
    ///
    /// Returns `None` if `rel ∉ [0, 1]`
    pub fn rel_to_freq(&self, rel: f64) -> Option<Frequency> {
        if !(0.0..=1.0).contains(&rel) {
            return None;
        }

        Some(self.range.start() + rel * self.range.signed_width())
    }

    /// Converts a relative coordinate in terms of the total width to a
    /// chemical shift.
    ///
    /// Returns `None` if `rel ∉ [0, 1]`
    pub fn rel_to_shift(&self, rel: f64) -> Option<Ratio> {
        if !(0.0..=1.0).contains(&rel) {
            return None;
        }

        let shift_range = self.shift_range();

        Some(shift_range.start() + rel * shift_range.signed_width())
    }

    /// Returns an iterator over `size` equally spaced frequencies from the
    /// start to the end of the frequency range.
    ///
    /// If `size <= 1`, the returned iterator does not contain any meaningful
    /// values.
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// will not be exactly identical to the end of the frequency range.
    ///
    /// Each call to this method will recompute the frequency values on the fly.
    pub fn freqs(&self, size: usize) -> impl Iterator<Item = Frequency> + use<> {
        let step = self.freq_step(size);
        let start = self.range.start();

        (0..size)
            .into_iter()
            .map(move |i| start + step * i as f64)
    }

    /// Returns a parallel iterator over `size` equally spaced frequencies from
    /// the start to the end of the frequency range.
    ///
    /// If `size <= 1`, the returned iterator does not contain any meaningful
    /// values.
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// will not be exactly identical to the end of the frequency range.
    ///
    /// Each call to this method will recompute the frequency values on the fly.
    #[cfg(feature = "rayon")]
    pub fn par_freqs(&self, size: usize) -> impl IndexedParallelIterator<Item = Frequency> + use<> {
        let step = self.freq_step(size);
        let start = self.range.start();

        (0..size)
            .into_par_iter()
            .map(move |i| start + step * i as f64)
    }

    /// Returns an iterator over `size` equally spaced chemical shifts from the
    /// start to the end of the chemical shift range.
    ///
    /// If `size <= 1`, the returned iterator does not contain any meaningful
    /// values.
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// will not be exactly identical to the end of the frequency range.
    ///
    /// Each call to this method will recompute the frequency values on the fly.
    pub fn shifts(&self, size: usize) -> impl Iterator<Item = Ratio> + use<> {
        let step = self.shift_step(size);
        let start = self.shift_range().start();

        (0..size)
            .into_iter()
            .map(move |i| start + step * i as f64)
    }

    /// Returns a parallel iterator over `size` equally spaced chemical shifts
    /// from the start to the end of the chemical shift range.
    ///
    /// If `size <= 1`, the returned iterator does not contain any meaningful
    /// values.
    ///
    /// Due to floating point errors when adding and multiplying, the end value
    /// will not be exactly identical to the end of the frequency range.
    ///
    /// Each call to this method will recompute the frequency values on the fly.
    #[cfg(feature = "rayon")]
    pub fn par_shifts(&self, size: usize) -> impl IndexedParallelIterator<Item = Ratio> + use<> {
        let step = self.shift_step(size);
        let start = self.shift_range().start();

        (0..size)
            .into_par_iter()
            .map(move |i| start + step * i as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;
    use uom::si::frequency::{hertz, megahertz};

    fn test_parameters() -> (FrequencyRange, Frequency, ShiftReference) {
        let range =
            FrequencyRange::new(Frequency::zero(), Frequency::new::<hertz>(12000.0)).unwrap();
        let larmor = Frequency::new::<megahertz>(600.25);
        let reference = ShiftReference::from_freq(Frequency::new::<hertz>(3000.0)).unwrap();

        (range, larmor, reference)
    }

    #[test]
    fn thread_safety() {
        assert_impl_all!(Axis: Send, Sync);
    }

    #[test]
    fn zero_width() {
        let range = FrequencyRange::new(Frequency::zero(), Frequency::zero()).unwrap();
        let (_, larmor, reference) = test_parameters();

        assert!(Axis::new(range, larmor, reference).is_none());
    }

    #[test]
    fn zero_larmor() {
        let larmor = Frequency::zero();
        let (range, _, reference) = test_parameters();

        assert!(Axis::new(range, larmor, reference).is_none());
    }

    #[test]
    fn negative_larmor() {
        let larmor = Frequency::new::<megahertz>(-700.0);
        let (range, _, reference) = test_parameters();

        assert!(Axis::new(range, larmor, reference).is_none());
    }

    #[test]
    fn non_finite_larmor() {
        let larmors = [
            Frequency::new::<hertz>(f64::NAN),
            Frequency::new::<hertz>(f64::INFINITY),
            Frequency::new::<hertz>(f64::NEG_INFINITY),
        ];
        let (range, _, reference) = test_parameters();

        larmors
            .into_iter()
            .for_each(|larmor| assert!(Axis::new(range, larmor, reference).is_none()));
    }

    macro_rules! axis_tests {
        ($name:ident, $range:expr, $larmor:expr, $reference:expr) => {
            mod $name {
                use super::*;
                use float_cmp::assert_approx_eq;
                use uom::si::ratio::part_per_million as ppm;

                #[test]
                fn step_size() {
                    let range = $range;
                    let larmor = $larmor;
                    let reference = $reference;
                    let axis = Axis::new(range, larmor, reference).unwrap();
                    let sizes = std::iter::once(0).chain((0..18).map(|i| 2_usize.pow(i)));

                    sizes.for_each(|size| match size {
                        0 | 1 => {
                            assert!(!axis.freq_step(size).is_nan());
                            assert!(!axis.shift_step(size).is_nan());
                        }
                        _ => {
                            let freq_range = axis.freq_range();
                            let freq_step = axis.freq_step(size);
                            let shift_range = axis.shift_range();
                            let shift_step = axis.shift_step(size);

                            assert_approx_eq!(
                                f64,
                                (freq_range.start() + freq_step * (size - 1) as f64).get::<hertz>(),
                                freq_range.end().get::<hertz>(),
                                epsilon = 1e-12
                            );
                            assert_approx_eq!(
                                f64,
                                (shift_range.start() + shift_step * (size - 1) as f64).get::<ppm>(),
                                shift_range.end().get::<ppm>(),
                                epsilon = 1e-12
                            );
                        }
                    })
                }

                #[test]
                fn freq_to_shift() {
                    let range = $range;
                    let larmor = $larmor;
                    let reference = $reference;
                    let axis = Axis::new(range, larmor, reference).unwrap();
                    let shift_range = axis.shift_range();
                    let size = 2_usize.pow(4);

                    assert_approx_eq!(
                        f64,
                        axis.freq_to_shift(range.start())
                            .unwrap()
                            .get::<ppm>(),
                        shift_range.start().get::<ppm>()
                    );
                    assert_approx_eq!(
                        f64,
                        axis.freq_to_shift(range.end())
                            .unwrap()
                            .get::<ppm>(),
                        shift_range.end().get::<ppm>()
                    );
                    // the converted values accumulate floating point errors much faster
                    // than the direct computations.
                    axis.freqs(size)
                        .map(|freq| axis.freq_to_shift(freq).unwrap())
                        .zip(axis.shifts(size))
                        .for_each(|(converted, computed)| {
                            assert_approx_eq!(
                                f64,
                                converted.get::<ppm>(),
                                computed.get::<ppm>(),
                                epsilon = 1e-12
                            );
                        });
                }

                #[test]
                fn to_rel() {
                    let range = $range;
                    let larmor = $larmor;
                    let reference = $reference;
                    let axis = Axis::new(range, larmor, reference).unwrap();
                    let shift_range = axis.shift_range();
                    let size = 2_usize.pow(4);

                    assert_approx_eq!(f64, axis.freq_to_rel(range.start()).unwrap(), 0.0);
                    assert_approx_eq!(f64, axis.freq_to_rel(range.end()).unwrap(), 1.0);
                    assert_approx_eq!(f64, axis.shift_to_rel(shift_range.start()).unwrap(), 0.0);
                    assert_approx_eq!(f64, axis.shift_to_rel(shift_range.end()).unwrap(), 1.0);

                    let freq_rels = axis
                        .freqs(size)
                        .take(size - 1)
                        .skip(1)
                        .map(|freq| axis.freq_to_rel(freq).unwrap());
                    let shift_rels = axis
                        .shifts(size)
                        .take(size - 1)
                        .skip(1)
                        .map(|shift| axis.shift_to_rel(shift).unwrap());

                    freq_rels
                        .zip(shift_rels)
                        .zip(
                            (1..size - 1)
                                .into_iter()
                                .map(|i| (i as f64) / ((size - 1) as f64)),
                        )
                        .for_each(|((freq_converted, shift_converted), computed)| {
                            assert_approx_eq!(f64, computed, freq_converted);
                            assert_approx_eq!(f64, computed, shift_converted);
                        });
                }

                #[test]
                fn rel_to() {
                    let range = $range;
                    let larmor = $larmor;
                    let reference = $reference;
                    let axis = Axis::new(range, larmor, reference).unwrap();
                    let shift_range = axis.shift_range();
                    let size = 2_usize.pow(4);

                    assert_approx_eq!(
                        f64,
                        axis.rel_to_freq(0.0).unwrap().get::<hertz>(),
                        range.start().get::<hertz>()
                    );
                    assert_approx_eq!(
                        f64,
                        axis.rel_to_freq(1.0).unwrap().get::<hertz>(),
                        range.end().get::<hertz>()
                    );
                    assert_approx_eq!(
                        f64,
                        axis.rel_to_shift(0.0).unwrap().get::<ppm>(),
                        shift_range.start().get::<ppm>()
                    );
                    assert_approx_eq!(
                        f64,
                        axis.rel_to_shift(1.0).unwrap().get::<ppm>(),
                        shift_range.end().get::<ppm>()
                    );
                    // the converted values accumulate floating point errors much faster
                    // than the direct computations.
                    (0..size)
                        .into_iter()
                        .map(|i| {
                            axis.rel_to_freq((i as f64) / ((size - 1) as f64))
                                .unwrap()
                        })
                        .zip(axis.freqs(size))
                        .for_each(|(converted, computed)| {
                            assert_approx_eq!(
                                f64,
                                converted.get::<hertz>(),
                                computed.get::<hertz>(),
                                epsilon = 1e-12
                            );
                        });
                    (0..size)
                        .into_iter()
                        .map(|i| {
                            axis.rel_to_shift((i as f64) / ((size - 1) as f64))
                                .unwrap()
                        })
                        .zip(axis.shifts(size))
                        .for_each(|(converted, computed)| {
                            assert_approx_eq!(
                                f64,
                                converted.get::<ppm>(),
                                computed.get::<ppm>(),
                                epsilon = 1e-12
                            );
                        });
                }
            }
        };
    }

    axis_tests!(
        forward_range_zero_reference,
        FrequencyRange::new(Frequency::zero(), Frequency::new::<hertz>(12000.0)).unwrap(),
        Frequency::new::<megahertz>(600.25),
        ShiftReference::from_freq(Frequency::new::<hertz>(3000.0)).unwrap()
    );
    axis_tests!(
        backward_range_zero_reference,
        FrequencyRange::new(Frequency::new::<hertz>(12000.0), Frequency::zero()).unwrap(),
        Frequency::new::<megahertz>(600.25),
        ShiftReference::from_freq(Frequency::new::<hertz>(3000.0)).unwrap()
    );
    axis_tests!(
        forward_range_non_zero_reference,
        FrequencyRange::new(Frequency::zero(), Frequency::new::<hertz>(12000.0)).unwrap(),
        Frequency::new::<megahertz>(600.25),
        ShiftReference::from_shift(Ratio::new::<ppm>(14.5)).unwrap()
    );
    axis_tests!(
        backward_range_non_zero_reference,
        FrequencyRange::new(Frequency::new::<hertz>(12000.0), Frequency::zero()).unwrap(),
        Frequency::new::<megahertz>(600.25),
        ShiftReference::from_shift(Ratio::new::<ppm>(14.5)).unwrap()
    );
}
