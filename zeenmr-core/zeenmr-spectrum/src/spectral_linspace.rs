use crate::error::{Error, Result};
use crate::{ChemicalShiftRange, FrequencyRange, ReferencingMethod, ShiftReference};
use std::ops::RangeBounds;
use uom::si::f64::{Frequency, Ratio};
use uom::si::frequency::hertz;
use uom::si::ratio::ratio;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a linear space of the spectral axis in a 1D NMR spectrum.
///
/// Converts raw NMR data into the known chemical shift scale and provides
/// iterators over frequency and chemical shift values.
///
/// # Design
///
/// Readjusting the chemical shift reference is a common operation in NMR data
/// analysis, so we only store the information necessary to compute frequency
/// and chemical shift values on the fly, rather than caching them in memory.
///
/// # Serialization with [Serde]
///
/// [Serde]: (https://serde.rs/)
///
/// When the `serde` feature is enabled, `SpectralLinspace` implements the
/// [`Serialize`] and [`Deserialize`] traits. This struct is flattened into
/// the [`Spectrum`] struct when serialized.
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
/// [`Spectrum`]: crate::Spectrum
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub(crate) struct SpectralLinspace {
    /// Larmor frequency of the nucleus used to convert frequency values to
    /// chemical shifts.
    ///
    /// Typical order of magnitude is 1e6 Hz. Also known as operating frequency
    /// and spectrometer frequency.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    larmor: Frequency,
    /// Frequency range of the spectrum.
    range: FrequencyRange,
    /// Number of points in the spectral axis.
    size: usize,
    /// Chemical shift reference.
    reference: ShiftReference,
}

impl SpectralLinspace {
    /// Constructs a new `SpectralLinspace`.
    ///
    /// # Errors
    ///
    /// This function will return an error if the chemical shift reference index
    /// is out of bounds for the given size, or if the frequency range or
    /// larmor frequency contains non-finite values, or if the larmor frequency
    /// is zero (<= [`f64::EPSILON`]).
    pub(crate) fn new<R, T>(larmor: Frequency, range: R, size: usize, reference: T) -> Result<Self>
    where
        R: Into<FrequencyRange>,
        T: Into<ShiftReference>,
    {
        let range = range.into();
        let reference = reference.into();

        Self::validate_range(range)?;
        Self::validate_larmor(larmor)?;
        Self::validate_reference(&reference, size)?;

        Ok(Self {
            larmor,
            range,
            size,
            reference,
        })
    }

    /// Returns the larmor frequency.
    pub(crate) fn larmor(&self) -> Frequency {
        self.larmor
    }

    /// Returns the number of points in the spectral axis.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Returns the chemical shift reference.
    pub(crate) fn shift_reference(&self) -> &ShiftReference {
        &self.reference
    }

    /// Calculates the offset from the chemical shift reference in ppm.
    pub(crate) fn reference_offset(&self) -> Ratio {
        self.reference.shift() - self.reference.index() as f64 * self.shift_step()
    }

    /// Returns the frequency range of the spectral axis.
    pub(crate) fn freq_range(&self) -> FrequencyRange {
        self.range
    }

    /// Returns the chemical shift range of the spectral axis.
    pub(crate) fn shift_range(&self) -> ChemicalShiftRange {
        let start = self.reference_offset();

        (start, start + self.width_ppm()).into()
    }

    /// Returns the width of the spectral axis in terms of frequency.
    pub(crate) fn freq_width(&self) -> Frequency {
        (self.range.end - self.range.start).abs()
    }

    /// Returns the width of the spectral axis in terms of chemical shift.
    pub(crate) fn width_ppm(&self) -> Ratio {
        self.freq_width() / self.larmor
    }

    /// Returns the central frequency of the spectral axis.
    pub(crate) fn freq_center(&self) -> Frequency {
        (self.range.start + self.range.end) / 2.0
    }

    /// Returns the central chemical shift of the spectral axis.
    pub(crate) fn shift_center(&self) -> Ratio {
        let range = self.shift_range();

        (range.start + range.end) / 2.0
    }

    /// Returns the step size of the spectral axis in terms of frequency.
    pub(crate) fn freq_step(&self) -> Frequency {
        (self.range.end - self.range.start) / (self.size as f64 - 1.0)
    }

    /// Returns the step size of the spectral axis in terms of chemical shift.
    pub(crate) fn shift_step(&self) -> Ratio {
        self.freq_step() / self.larmor
    }

    /// Returns the step size of the spectral axis in relative units.
    pub(crate) fn relative_step(&self) -> f64 {
        1.0 / (self.size as f64 - 1.0)
    }

    /// Calculates the fractional index of a frequency within the linear space.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency is out of bounds of the spectral axis
    /// or non-finite.
    pub(crate) fn freq_to_fractional(&self, frequency: Frequency) -> Result<f64> {
        match (frequency.is_finite(), self.contains_freq(frequency)) {
            (true, true) => Ok(((frequency - self.range.start) * ((self.size - 1) as f64)
                / (self.range.end - self.range.start))
                .get::<ratio>()),
            (false, _) => Err(Error::non_finite_float()),
            (_, false) => Err(Error::out_of_bounds()),
        }
    }

    /// Calculates the fractional index of a chemical shift within the linear
    /// space.
    ///
    /// # Errors
    ///
    /// Returns an error if the chemical shift is out of bounds of the spectral
    /// axis or non-finite.
    pub(crate) fn shift_to_fractional(&self, shift: Ratio) -> Result<f64> {
        match (shift.is_finite(), self.contains_shift(shift)) {
            (true, true) => Ok(((shift - self.reference_offset())
                * ((self.size - 1) as f64)
                * self.larmor
                / (self.range.end - self.range.start))
                .get::<ratio>()),
            (false, _) => Err(Error::non_finite_float()),
            (_, false) => Err(Error::out_of_bounds()),
        }
    }

    /// Calculates the fractional index of a fraction of the linear space.
    ///
    /// # Errors
    ///
    /// Returns an error if the relative value is not between 0 and 1,
    /// inclusive, or non-finite.
    pub(crate) fn relative_to_fractional(&self, relative: f64) -> Result<f64> {
        match (relative.is_finite(), (0.0..=1.0).contains(&relative)) {
            (true, true) => Ok(relative * ((self.size - 1) as f64)),
            (false, _) => Err(Error::non_finite_float()),
            (_, false) => Err(Error::out_of_bounds()),
        }
    }

    /// Converts an index within the linear space to a frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn index_to_freq(&self, index: usize) -> Result<Frequency> {
        Self::validate_index(index, self.size)?;

        Ok(self.range.start + self.freq_step() * index as f64)
    }

    /// Converts an index within the linear space to a chemical shift.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn index_to_shift(&self, index: usize) -> Result<Ratio> {
        Self::validate_index(index, self.size)?;
        let step = self.shift_step();
        let offset = self.reference_offset();

        Ok(offset + step * index as f64)
    }

    /// Converts an index within the linear space to a relative value.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn index_to_relative(&self, index: usize) -> Result<f64> {
        Self::validate_index(index, self.size)?;

        Ok(self.relative_step() * index as f64)
    }

    /// Checks if the given frequency is within the linear space.
    pub(crate) fn contains_freq(&self, frequency: Frequency) -> bool {
        frequency.is_finite() && self.range.ordered().contains(&frequency)
    }

    /// Checks if the given chemical shift in ppm is within the linear space.
    pub(crate) fn contains_shift(&self, shift: Ratio) -> bool {
        shift.is_finite() && self.shift_range().ordered().contains(&shift)
    }

    /// Returns an iterator over the frequencies.
    ///
    /// Computing each frequency value only requires one addition and one
    /// multiplication, so we opt not to cache the frequencies in memory.
    pub(crate) fn frequencies(&self) -> impl Iterator<Item = Frequency> + use<> {
        let step = self.freq_step();
        let start = self.range.start;

        (0..self.size).map(move |i| start + step * i as f64)
    }

    /// Returns an iterator over the chemical shifts in ppm.
    ///
    /// Computing each chemical shift value only requires one addition and one
    /// multiplication, so we opt not to cache the chemical shifts in memory.
    pub(crate) fn shifts(&self) -> impl Iterator<Item = Ratio> + use<> {
        let step = self.shift_step();
        let offset = self.reference_offset();

        (0..self.size).map(move |i| offset + step * i as f64)
    }

    /// Sets the frequency range of the spectrum.
    ///
    /// The order of the range determines the direction of the spectral axis,
    /// where the first value is the first point, and the second value is the
    /// last point.
    ///
    /// # Errors
    ///
    /// Returns an error if either frequency in the range is not a finite float.
    pub(crate) fn set_range<R>(&mut self, range: R) -> Result<()>
    where
        R: Into<FrequencyRange>,
    {
        let range = range.into();
        Self::validate_range(range)?;
        self.range = range;

        Ok(())
    }

    /// Sets the larmor frequency.
    ///
    /// This value is used to standardize the chemical shifts in the spectrum.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    ///
    /// # Errors
    ///
    /// Returns an error if the larmor frequency is not a finite float or
    /// zero (<= [`f64::EPSILON`]).
    pub(crate) fn set_larmor(&mut self, larmor: Frequency) -> Result<()> {
        Self::validate_larmor(larmor)?;
        self.larmor = larmor;

        Ok(())
    }

    /// Sets the chemical shift reference.
    ///
    /// # Errors
    ///
    /// Returns an error if the chemical shift value is not a finite float or
    /// the new reference index is out of bounds for the current size of the
    /// spectral axis.
    pub(crate) fn set_shift_reference<T>(&mut self, reference: T) -> Result<()>
    where
        T: Into<ShiftReference>,
    {
        let reference = reference.into();
        Self::validate_reference(&reference, self.size)?;
        self.reference = reference;

        Ok(())
    }

    /// Sets the chemical shift reference value in ppm.
    ///
    /// # Errors
    ///
    /// Returns an error if the chemical shift is not a finite float.
    pub(crate) fn set_shift_reference_value(&mut self, shift: Ratio) -> Result<()> {
        Self::validate_shift_value(shift)?;
        self.reference.set_shift(shift);

        Ok(())
    }

    /// Sets the chemical shift reference index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn set_shift_reference_index(&mut self, index: usize) -> Result<()> {
        Self::validate_index(index, self.size)?;
        self.reference.set_index(index);

        Ok(())
    }

    /// Sets the chemical shift reference name.
    pub(crate) fn set_shift_reference_name<T>(&mut self, name: T)
    where
        T: Into<String>,
    {
        self.reference.set_name(name);
    }

    /// Clears the chemical shift reference name.
    pub(crate) fn clear_shift_reference_name(&mut self) {
        self.reference.clear_name();
    }

    /// Sets the chemical shift reference method.
    pub(crate) fn set_shift_reference_method<T>(&mut self, method: T)
    where
        T: Into<ReferencingMethod>,
    {
        self.reference.set_method(method);
    }

    /// Clears the chemical shift reference method.
    pub(crate) fn clear_shift_reference_method(&mut self) {
        self.reference.clear_method();
    }

    /// Validates the frequency range and returns an error if either frequency
    /// is not a finite float or not greater than or equal to 0.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidRange`](crate::error::Kind::InvalidRange)
    fn validate_range(range: FrequencyRange) -> Result<()> {
        match range.start.is_finite() && range.end.is_finite() {
            true => Ok(()),
            false => Err(Error::invalid_range()),
        }
    }

    /// Validates the larmor frequency and returns an error if it is too close
    /// to zero (<= [`f64::EPSILON`]) or not a finite float.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidSpectrometerFrequency`](crate::error::Kind::InvalidLarmor)
    fn validate_larmor(larmor: Frequency) -> Result<()> {
        match (
            larmor.is_finite(),
            larmor.abs() > Frequency::new::<hertz>(f64::EPSILON),
        ) {
            (true, true) => Ok(()),
            (false, _) => Err(Error::invalid_larmor(Some(Error::non_finite_float()))),
            (_, false) => Err(Error::invalid_larmor(None)),
        }
    }

    /// Validates the chemical shift value and returns an error if it is not a
    /// finite float.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`NonFiniteFloat`](crate::error::Kind::NonFiniteFloat)
    fn validate_shift_value(shift: Ratio) -> Result<()> {
        match shift.is_finite() {
            true => Ok(()),
            false => Err(Error::non_finite_float()),
        }
    }

    /// Validates a provided index and returns an error if it is out of bounds
    /// for the given size.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`OutOfBounds`](crate::error::Kind::OutOfBounds)
    fn validate_index(index: usize, size: usize) -> Result<()> {
        match index < size {
            true => Ok(()),
            false => Err(Error::out_of_bounds()),
        }
    }

    /// Validates the shift reference and returns an error if the index is out
    /// of bounds or the chemical shift is not finite.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidShiftReference`](crate::error::Kind::InvalidShiftReference)
    fn validate_reference(reference: &ShiftReference, size: usize) -> Result<()> {
        match Self::validate_index(reference.index(), size) {
            Ok(_) => match Self::validate_shift_value(reference.shift()) {
                Ok(_) => Ok(()),
                Err(error) => Err(Error::invalid_shift_reference(error)),
            },
            Err(error) => Err(Error::invalid_shift_reference(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Kind;
    use float_cmp::assert_approx_eq;
    use num_traits::Zero;
    use static_assertions::assert_impl_all;
    use uom::si::frequency::{hertz, megahertz};
    use uom::si::ratio::part_per_million as ppm;

    fn valid_parameters() -> (Frequency, (Frequency, Frequency), usize, ShiftReference) {
        let larmor = Frequency::new::<megahertz>(600.0);
        let range = (Frequency::new::<hertz>(12000.0), Frequency::zero());
        let size = 2_usize.pow(17);
        let reference = (range.0 / larmor).into();

        (larmor, range, size, reference)
    }

    fn valid_linspace() -> SpectralLinspace {
        let (larmor, range, size, reference) = valid_parameters();

        SpectralLinspace::new(larmor, range, size, reference).unwrap()
    }

    #[test]
    fn thread_safety() {
        assert_impl_all!(SpectralLinspace: Send, Sync);
    }

    #[test]
    fn new() {
        let (larmor, range, size, reference) = valid_parameters();
        let linspace = SpectralLinspace::new(larmor, range, size, reference);
        assert!(linspace.is_ok());
    }

    #[test]
    fn invalid_frequency_range() {
        let (larmor, _, size, reference) = valid_parameters();
        let invalid_ranges = [
            (Frequency::new::<hertz>(f64::NAN), Frequency::zero()),
            (Frequency::new::<hertz>(f64::INFINITY), Frequency::zero()),
        ];
        let errors = invalid_ranges.map(|range| {
            SpectralLinspace::new(larmor, range, size, reference.clone()).unwrap_err()
        });
        let expected_sources = [
            None,
            None,
            Some(Error::non_finite_float()),
            Some(Error::non_finite_float()),
        ];
        errors
            .into_iter()
            .zip(expected_sources.clone())
            .for_each(|(error, source)| match error.kind() {
                Kind::InvalidRange => assert_eq!(error.source(), source.as_ref()),
                _ => panic!("unexpected error: {error}"),
            });

        let mut linspace = valid_linspace();
        invalid_ranges
            .iter()
            .zip(expected_sources)
            .for_each(|(range, source)| {
                assert_eq!(
                    linspace.set_range(*range).unwrap_err().source(),
                    source.as_ref()
                );
            });
    }

    #[test]
    fn invalid_larmor() {
        let (_, range, size, reference) = valid_parameters();
        let invalid_larmor = [
            Frequency::zero(),
            Frequency::new::<megahertz>(f64::NAN),
            Frequency::new::<megahertz>(f64::INFINITY),
            Frequency::new::<megahertz>(f64::NEG_INFINITY),
        ];
        let errors = invalid_larmor.map(|larmor| {
            SpectralLinspace::new(larmor, range, size, reference.clone()).unwrap_err()
        });
        let expected_sources = [
            None,
            Some(Error::non_finite_float()),
            Some(Error::non_finite_float()),
            Some(Error::non_finite_float()),
        ];
        errors
            .into_iter()
            .zip(expected_sources.clone())
            .for_each(|(error, source)| match error.kind() {
                Kind::InvalidLarmor => assert_eq!(error.source(), source.as_ref()),
                _ => panic!("unexpected error: {error}"),
            });

        let mut linspace = valid_linspace();
        invalid_larmor
            .iter()
            .zip(expected_sources)
            .for_each(|(freq, source)| {
                assert_eq!(
                    linspace.set_larmor(*freq).unwrap_err().source(),
                    source.as_ref()
                );
            });
    }

    #[test]
    fn index_out_of_bounds() {
        let linspace = valid_linspace();
        let errors = [
            linspace
                .index_to_freq(2_usize.pow(18))
                .unwrap_err(),
            linspace
                .index_to_shift(2_usize.pow(18))
                .unwrap_err(),
            linspace
                .index_to_relative(2_usize.pow(18))
                .unwrap_err(),
        ];
        errors
            .into_iter()
            .for_each(|error| assert_eq!(error, Error::out_of_bounds()));
    }

    #[test]
    fn invalid_shift_reference() {
        let (larmor, range, size, _) = valid_parameters();
        let invalid_references = [
            ShiftReference::from((range.0 / larmor, size)),
            ShiftReference::from((Ratio::new::<ratio>(f64::NAN), size / 2)),
        ];
        let errors = invalid_references
            .clone()
            .map(|reference| SpectralLinspace::new(larmor, range, size, reference).unwrap_err());
        let expected_sources = [
            Some(Error::out_of_bounds()),
            Some(Error::non_finite_float()),
        ];
        errors
            .into_iter()
            .zip(expected_sources)
            .for_each(|(error, source)| match error.kind() {
                Kind::InvalidShiftReference => assert_eq!(error.source(), source.as_ref()),
                _ => panic!("unexpected error: {error}"),
            })
    }

    #[test]
    fn mutators() {
        let mut linspace = valid_linspace();
        assert!(
            linspace
                .set_range((Frequency::new::<hertz>(24000.0), Frequency::zero()))
                .is_ok()
        );
        assert_approx_eq!(f64, linspace.freq_range().start.get::<hertz>(), 24000.0);
        assert_approx_eq!(f64, linspace.freq_range().end.get::<hertz>(), 0.0);
        assert!(
            linspace
                .set_larmor(Frequency::new::<megahertz>(800.0))
                .is_ok()
        );
        assert_approx_eq!(f64, linspace.larmor().get::<megahertz>(), 800.0);
        assert!(
            linspace
                .set_shift_reference((Ratio::new::<ppm>(24000.0 / 800.0), 0))
                .is_ok()
        );
        assert_approx_eq!(
            f64,
            linspace.shift_reference().shift().get::<ppm>(),
            24000.0 / 800.0
        );
        assert_eq!(linspace.shift_reference().index(), 0);
        assert!(linspace.shift_reference().name().is_none());
        assert!(linspace.shift_reference().method().is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let larmor = Frequency::new::<megahertz>(600.25);
        let range = (Frequency::new::<hertz>(12000.0), Frequency::zero());
        let size = 2_usize.pow(15);
        let reference = range.0 / larmor;
        let linspace = SpectralLinspace::new(larmor, range, size, reference).unwrap();
        let serialized = serde_json::to_string(&linspace).unwrap();
        let deserialized = serde_json::from_str::<SpectralLinspace>(&serialized).unwrap();
        assert_approx_eq!(
            f64,
            linspace.range.start.get::<hertz>(),
            deserialized.range.start.get::<hertz>()
        );
        assert_approx_eq!(
            f64,
            linspace.range.end.get::<hertz>(),
            deserialized.range.end.get::<hertz>()
        );
        assert_approx_eq!(
            f64,
            linspace.larmor.get::<megahertz>(),
            deserialized.larmor.get::<megahertz>()
        );
        assert_eq!(linspace.size, deserialized.size);
        assert_approx_eq!(
            f64,
            linspace.reference.shift().get::<ppm>(),
            deserialized.reference.shift().get::<ppm>()
        );
        assert_eq!(linspace.reference.index(), deserialized.reference.index());
        assert_eq!(linspace.reference.name(), deserialized.reference.name());
        assert_eq!(linspace.reference.method(), deserialized.reference.method());
    }
}
