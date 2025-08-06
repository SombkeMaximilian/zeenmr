use crate::ShiftReference;
use crate::error::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a linear space of the spectral axis in a 1D NMR spectrum.
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
    /// Spectrometer frequency in MHz.
    spectrometer_frequency: f64,
    /// Frequency range of the spectrum in Hz.
    frequency_range: (f64, f64),
    /// Number of points in the spectral axis.
    size: usize,
    /// Chemical shift reference.
    reference: ShiftReference,
}

impl SpectralLinspace {
    /// Constructs a new `SpectralLinspace`.
    ///
    /// The first value of the `frequency_range` is the frequency of the first
    /// point in the spectral axis, while the second value is the last point.
    /// The `spectrometer_frequency` is used to standardize the chemical
    /// shifts in the spectrum. A `ShiftReference` is used to determine the
    /// chemical shift reference point in the spectrum.
    ///
    /// # Errors
    ///
    /// This function will return an error if the chemical shift reference index
    /// is out of bounds for the given size, or if the frequency range or
    /// spectrometer frequency contains non-finite values.
    pub(crate) fn new<T: Into<ShiftReference>>(
        spectrometer_frequency: f64,
        frequency_range: (f64, f64),
        size: usize,
        reference: T,
    ) -> Result<Self> {
        let reference = reference.into();

        Self::validate_frequency_range(frequency_range)?;
        Self::validate_spectrometer_frequency(spectrometer_frequency)?;
        Self::validate_index(reference.index(), size)?;

        Ok(Self {
            spectrometer_frequency,
            frequency_range,
            size,
            reference,
        })
    }

    /// Returns the spectrometer frequency in MHz.
    pub(crate) fn spectrometer_frequency(&self) -> f64 {
        self.spectrometer_frequency
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
    pub(crate) fn reference_offset(&self) -> f64 {
        self.reference.chemical_shift() - self.reference.index() as f64 * self.step_ppm()
    }

    /// Returns the frequency range of the spectral axis in Hz.
    pub(crate) fn range_hz(&self) -> (f64, f64) {
        self.frequency_range
    }

    /// Returns the chemical shift range of the spectral axis in ppm.
    pub(crate) fn range_ppm(&self) -> (f64, f64) {
        let start = self.reference_offset();

        (
            start,
            start + (self.frequency_range.1 - self.frequency_range.0) / self.spectrometer_frequency,
        )
    }

    /// Returns the width of the spectral axis in Hz.
    pub(crate) fn width_hz(&self) -> f64 {
        f64::abs(self.frequency_range.1 - self.frequency_range.0)
    }

    /// Returns the width of the spectral axis in ppm.
    pub(crate) fn width_ppm(&self) -> f64 {
        self.width_hz() / self.spectrometer_frequency
    }

    /// Returns the center frequency of the spectral axis in Hz.
    pub(crate) fn center_hz(&self) -> f64 {
        (self.frequency_range.0 + self.frequency_range.1) / 2.0
    }

    /// Returns the center frequency of the spectral axis in ppm.
    pub(crate) fn center_ppm(&self) -> f64 {
        let range = self.range_ppm();

        (range.1 + range.0) / 2.0
    }

    /// Returns the step size of the spectral axis in Hz.
    pub(crate) fn step_hz(&self) -> f64 {
        (self.frequency_range.1 - self.frequency_range.0) / (self.size as f64 - 1.0)
    }

    /// Returns the step size of the spectral axis in ppm.
    pub(crate) fn step_ppm(&self) -> f64 {
        self.step_hz() / self.spectrometer_frequency
    }

    /// Returns the step size of the spectral axis in relative units.
    pub(crate) fn step_relative(&self) -> f64 {
        1.0 / (self.size as f64 - 1.0)
    }

    /// Calculates the fractional index of a frequency within the linear space.
    pub(crate) fn hz_to_fractional(&self, frequency: f64) -> f64 {
        (frequency - self.range_hz().0) / self.step_hz()
    }

    /// Calculates the fractional index of a chemical shift within the linear
    /// space.
    pub(crate) fn ppm_to_fractional(&self, chemical_shift: f64) -> f64 {
        (chemical_shift - self.range_ppm().0) / self.step_ppm()
    }

    /// Calculates the fractional index in relative units within the linear
    /// space.
    pub(crate) fn relative_to_fractional(&self, relative: f64) -> f64 {
        relative / self.step_relative()
    }

    /// Converts an index within the linear space to a frequency in Hz.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn index_to_hz(&self, index: usize) -> Result<f64> {
        Self::validate_index(index, self.size)?;

        Ok(self.frequency_range.0 + self.step_hz() * index as f64)
    }

    /// Converts an index within the linear space to a chemical shift in ppm.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds for the current size
    /// of the spectral axis.
    pub(crate) fn index_to_ppm(&self, index: usize) -> Result<f64> {
        Self::validate_index(index, self.size)?;
        let step = self.step_ppm();
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

        Ok(self.step_relative() * index as f64)
    }

    /// Checks if the given frequency in Hz is within the linear space.
    pub(crate) fn contains_hz(&self, frequency: f64) -> bool {
        let range = (
            f64::min(self.frequency_range.0, self.frequency_range.1),
            f64::max(self.frequency_range.0, self.frequency_range.1),
        );

        frequency.is_finite() && frequency >= range.0 && frequency <= range.1
    }

    /// Checks if the given chemical shift in ppm is within the linear space.
    pub(crate) fn contains_ppm(&self, chemical_shift: f64) -> bool {
        let range = self.range_ppm();
        let range = (f64::min(range.0, range.1), f64::max(range.0, range.1));

        chemical_shift.is_finite() && chemical_shift >= range.0 && chemical_shift <= range.1
    }

    /// Returns an iterator over the frequencies in Hz.
    ///
    /// Computing each frequency value only requires one addition and one
    /// multiplication, so we opt not to cache the frequencies in memory.
    pub(crate) fn frequencies(&self) -> impl Iterator<Item = f64> {
        let step = self.step_hz();

        (0..self.size).map(move |i| self.frequency_range.0 + step * i as f64)
    }

    /// Returns an iterator over the chemical shifts in ppm.
    ///
    /// Computing each chemical shift value only requires one addition and one
    /// multiplication, so we opt not to cache the chemical shifts in memory.
    pub(crate) fn chemical_shifts(&self) -> impl Iterator<Item = f64> {
        let step = self.step_ppm();
        let offset = self.reference_offset();

        (0..self.size).map(move |i| offset + step * i as f64)
    }

    /// Sets the frequency range of the spectrum in Hz.
    ///
    /// The order of the range determines the direction of the spectral axis,
    /// where the first value is the first point, and the second value is the
    /// last point.
    ///
    /// # Errors
    ///
    /// Returns an error if either frequency in the range is not a finite float.
    pub(crate) fn set_frequency_range(&mut self, frequency_range: (f64, f64)) -> Result<()> {
        Self::validate_frequency_range(frequency_range)?;
        self.frequency_range = frequency_range;

        Ok(())
    }

    /// Sets the spectrometer frequency in MHz.
    ///
    /// This value is used to standardize the chemical shifts in the spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if the spectrometer frequency is not a finite float.
    pub(crate) fn set_spectrometer_frequency(&mut self, spectrometer_frequency: f64) -> Result<()> {
        Self::validate_spectrometer_frequency(spectrometer_frequency)?;
        self.spectrometer_frequency = spectrometer_frequency;

        Ok(())
    }

    /// Sets the chemical shift reference.
    ///
    /// # Errors
    ///
    /// Returns an error if the new reference index is out of bounds for the
    /// current size of the spectral axis.
    pub(crate) fn set_shift_reference<T: Into<ShiftReference>>(
        &mut self,
        reference: T,
    ) -> Result<()> {
        let reference = reference.into();
        Self::validate_index(reference.index(), self.size)?;
        self.reference = reference;

        Ok(())
    }

    /// Internal helper function to validate the frequency range and return an
    /// error if either frequency is not a finite float.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidFrequencyRange`]: crate::error::Kind::InvalidFrequencyRange
    fn validate_frequency_range(frequency_range: (f64, f64)) -> Result<()> {
        match frequency_range.0.is_finite() && frequency_range.1.is_finite() {
            true => Ok(()),
            false => Err(Error::invalid_frequency_range(frequency_range)),
        }
    }

    /// Internal helper function to validate the spectrometer frequency and
    /// return an error if it is not a finite float.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidSpectrometerFrequency`]:
    ///   crate::error::Kind::InvalidSpectrometerFrequency
    fn validate_spectrometer_frequency(spectrometer_frequency: f64) -> Result<()> {
        match spectrometer_frequency.is_finite() {
            true => Ok(()),
            false => Err(Error::invalid_spectrometer_frequency(
                spectrometer_frequency,
            )),
        }
    }

    /// Internal helper function to validate the reference index and return an
    /// error if it is out of bounds for the given size.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`IndexOutOfBounds`]: crate::error::Kind::IndexOutOfBounds
    fn validate_index(index: usize, size: usize) -> Result<()> {
        match index < size {
            true => Ok(()),
            false => Err(Error::index_out_of_bounds(index, size)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ShiftReference: Send, Sync);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let frequency_range = (12000.0, 0.0);
        let spectrometer_frequency = 600.25;
        let spectrum_size = 2_usize.pow(15);
        let first_point = frequency_range.0 / spectrometer_frequency;
        let linspace = SpectralLinspace::new(
            spectrometer_frequency,
            frequency_range,
            spectrum_size,
            first_point,
        )
        .unwrap();
        let serialized = serde_json::to_string(&linspace).unwrap();
        let deserialized = serde_json::from_str::<SpectralLinspace>(&serialized).unwrap();
        assert_approx_eq!(
            f64,
            linspace.frequency_range.0,
            deserialized.frequency_range.0
        );
        assert_approx_eq!(
            f64,
            linspace.frequency_range.1,
            deserialized.frequency_range.1
        );
        assert_approx_eq!(
            f64,
            linspace.spectrometer_frequency,
            deserialized.spectrometer_frequency
        );
        assert_eq!(linspace.size, deserialized.size);
        assert_approx_eq!(
            f64,
            linspace.reference.chemical_shift(),
            deserialized.reference.chemical_shift()
        );
        assert_eq!(linspace.reference.index(), deserialized.reference.index());
        assert_eq!(linspace.reference.name(), deserialized.reference.name());
        assert_eq!(linspace.reference.method(), deserialized.reference.method());
    }

    #[test]
    fn properties() {
        let frequency_range = (12000.0, 0.0);
        let spectrometer_frequency = 600.0;
        let spectrum_size = 2_usize.pow(17);
        let first_point = frequency_range.0 / spectrometer_frequency;
        let linspace = SpectralLinspace::new(
            spectrometer_frequency,
            frequency_range,
            spectrum_size,
            first_point,
        )
        .unwrap();

        assert_approx_eq!(
            f64,
            linspace.spectrometer_frequency(),
            spectrometer_frequency
        );
        assert_eq!(linspace.size(), spectrum_size);
        assert_approx_eq!(f64, linspace.range_hz().0, frequency_range.0);
        assert_approx_eq!(f64, linspace.range_hz().1, frequency_range.1);
        assert_approx_eq!(f64, linspace.range_ppm().0, first_point);
        assert_approx_eq!(
            f64,
            linspace.range_ppm().1,
            first_point + (frequency_range.1 - frequency_range.0) / spectrometer_frequency
        );
        assert_approx_eq!(
            f64,
            linspace.width_hz(),
            f64::abs(frequency_range.1 - frequency_range.0)
        );
        assert_approx_eq!(
            f64,
            linspace.width_ppm(),
            f64::abs((frequency_range.1 - frequency_range.0) / spectrometer_frequency)
        );
        assert_approx_eq!(
            f64,
            linspace.center_hz(),
            (frequency_range.0 + frequency_range.1) / 2.0
        );
        assert_approx_eq!(
            f64,
            linspace.center_ppm(),
            first_point + (frequency_range.1 - frequency_range.0) / (2.0 * spectrometer_frequency)
        );
        assert_approx_eq!(
            f64,
            linspace.step_hz(),
            (frequency_range.1 - frequency_range.0) / (spectrum_size as f64 - 1.0)
        );
        assert_approx_eq!(
            f64,
            linspace.step_ppm(),
            (frequency_range.1 - frequency_range.0)
                / ((spectrum_size as f64 - 1.0) * spectrometer_frequency)
        );
    }
}
