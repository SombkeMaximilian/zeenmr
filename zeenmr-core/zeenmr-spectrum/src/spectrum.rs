use crate::error::{Error, Result};
use crate::{Nucleus, ShiftReference, SignalBoundaries, SpectralLinspace};
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data structure that represents a 1D NMR spectrum.
///
/// `Spectrum` is a container that holds the signal intensities and metadata of
/// a 1D NMR spectrum. The signal intensities are read-only, but the metadata
/// may be modified (e.g., the chemical shift reference).
///
/// # Invariants
///
/// A valid `Spectrum` instance maintains the following conditions:
/// - None of the values are [`f64::NAN`], [`f64::INFINITY`], or
///   [`f64::NEG_INFINITY`].
/// - The frequency range contains only values greater than or equal to zero.
/// - The spectrometer frequency is a positive value greater than zero.
/// - The index of the [`ShiftReference`] is within the bounds of the spectral
///   axis.
/// - Signal boundaries are within the spectral axis.
///
/// # Thread Safety
///
/// The `Spectrum` type is both [`Send`] and [`Sync`], meaning it can safely be
/// shared and accessed across threads. This makes `Spectrum` suitable for use
/// in concurrent or parallel applications.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Spectrum` implements the [`Serialize`]
/// and [`Deserialize`] traits.
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
/// [MessagePack]: https://docs.rs/rmp-serde
///
/// # Example: Constructing a `Spectrum` manually
///
/// The following demonstrates how to create a `Spectrum` object manually. This
/// is typically not how spectra are created, as they are usually parsed from
/// files. However, it can be useful for generating synthetic data or handling
/// custom formats.
///
/// ```
/// use zeenmr_spectrum::{ShiftReference, SignalBoundaries, Spectrum};
///
/// # fn main() -> zeenmr_spectrum::error::Result<()> {
/// // Define spectrometer frequency in MHz frequency range in Hz.
/// let spectrometer_frequency = 600.0;
/// let frequency_range = (0.0, 12000.0);
///
/// // Generate intensities using 3 Lorentzian peaks.
/// let intensities = (0..2_u32.pow(15))
///     .map(|i| i as f64 * frequency_range.1 / ((2_u32.pow(15) - 1) as f64))
///     .map(|f| f / spectrometer_frequency)
///     .map(|x| {
///         // Reference signal centered at 5 ppm.
///         10.0 * 0.25 / (0.15_f64.powi(2) + (x - 5.0).powi(2))
///             // The left signal is centered at 8 ppm.
///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 8.0).powi(2))
///             // The right signal is centered at 12 ppm.
///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 12.0).powi(2))
///     })
///     .collect::<Vec<f64>>();
///
/// // Create a Spectrum object.
/// let mut spectrum = Spectrum::new(intensities, spectrometer_frequency, frequency_range)?;
///
/// // Specify a chemical shift reference.
/// let shift_reference = ShiftReference::new(0.0, 2_usize.pow(13), Some("ref"), Some("internal"));
/// spectrum.set_shift_reference(shift_reference)?;
///
/// // Add metadata.
/// spectrum.set_id("example_spectrum");
/// spectrum.set_nucleus("Deuterium");
/// spectrum.set_signal_boundaries(SignalBoundaries::Relative(0.1, 0.9))?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct Spectrum {
    /// ID of the spectrum, if available.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none",)
    )]
    id: Option<String>,
    /// Nucleus observed in the NMR experiment.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    nucleus: Option<Nucleus>,
    /// Linear space of the spectral axis.
    #[cfg_attr(feature = "serde", serde(flatten))]
    spectral_linspace: SpectralLinspace,
    /// Boundaries of the signal region in the spectrum.
    signal_boundaries: (usize, usize),
    /// Intensity values in arbitrary units.
    #[cfg_attr(feature = "serde", serde(with = "serialize_intensities"))]
    intensities: Arc<[f64]>,
}

// Enabling the `rc` feature of serde would force it on everyone using this
// crate, so we just implement this manually for now. If `Arc` turns out to be
// unnecessary, we can switch to `Box<[f64]>` later and remove this module.
#[cfg(feature = "serde")]
mod serialize_intensities {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub(crate) fn serialize<S>(values: &Arc<[f64]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values.as_ref().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[f64]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Box::<[f64]>::deserialize(deserializer)?.into())
    }
}

impl Spectrum {
    /// Constructs a `Spectrum` from an iterator of intensities and parameters
    /// for the spectral axis.
    ///
    /// `frequency_range` is specified in Hz, and `spectrometer_frequency` is
    /// specified in MHz. Note that the `intensities` cannot be modified after
    /// construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the input data violates any of the invariants
    /// required for a valid `Spectrum`. The following conditions are checked:
    ///
    /// - Neither `spectrometer_frequency` nor `frequency_range` contain
    ///   non-finite values (no [`f64::NAN`], [`f64::INFINITY`] or
    ///   [`f64::NEG_INFINITY`]).
    /// - The `intensities` are not empty and do not contain any non-finite
    ///   values.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ShiftReference, SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// // Define spectrometer frequency in MHz frequency range in Hz.
    /// let spectrometer_frequency = 600.0;
    /// let frequency_range = (0.0, 12000.0);
    ///
    /// // Generate intensities using 3 Lorentzian peaks.
    /// let intensities = (0..2_u32.pow(15))
    ///     .map(|i| i as f64 * frequency_range.1 / ((2_u32.pow(15) - 1) as f64))
    ///     .map(|f| f / spectrometer_frequency)
    ///     .map(|x| {
    ///         // Reference signal centered at 5 ppm.
    ///         10.0 * 0.25 / (0.15_f64.powi(2) + (x - 5.0).powi(2))
    ///             // The left signal is centered at 8 ppm.
    ///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 8.0).powi(2))
    ///             // The right signal is centered at 12 ppm.
    ///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 12.0).powi(2))
    ///     })
    ///     .collect::<Vec<f64>>();
    ///
    /// // Create a Spectrum object.
    /// let mut spectrum = Spectrum::new(intensities, spectrometer_frequency, frequency_range)?;
    ///
    /// // Add metadata.
    /// spectrum.set_id("example_spectrum");
    /// spectrum.set_nucleus("Deuterium");
    /// spectrum.set_signal_boundaries(SignalBoundaries::Relative(0.1, 0.9))?;
    ///
    /// // Specify a chemical shift reference.
    /// let shift_reference = ShiftReference::new(0.0, 2_usize.pow(13), Some("TMS"), Some("internal"));
    /// spectrum.set_shift_reference(shift_reference)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<I>(
        intensities: I,
        spectrometer_frequency: f64,
        frequency_range: (f64, f64),
    ) -> Result<Self>
    where
        I: IntoIterator<Item = f64>,
    {
        let intensities = Self::validate_intensities(intensities)?;
        let reference = frequency_range.0 / spectrometer_frequency;
        let spectral_linspace = SpectralLinspace::new(
            spectrometer_frequency,
            frequency_range,
            intensities.len(),
            reference,
        )?;
        let signal_boundaries = Self::validate_boundaries(Default::default(), &spectral_linspace)?;

        Ok(Self {
            id: None,
            nucleus: None,
            spectral_linspace,
            signal_boundaries,
            intensities,
        })
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn nucleus(&self) -> Option<&Nucleus> {
        self.nucleus.as_ref()
    }

    pub fn spectrometer_frequency(&self) -> f64 {
        self.spectral_linspace.spectrometer_frequency()
    }

    pub fn len(&self) -> usize {
        self.spectral_linspace.size()
    }

    pub fn shift_reference(&self) -> &ShiftReference {
        self.spectral_linspace.shift_reference()
    }

    pub fn range_hz(&self) -> (f64, f64) {
        self.spectral_linspace.range_hz()
    }

    pub fn range_ppm(&self) -> (f64, f64) {
        self.spectral_linspace.range_ppm()
    }

    pub fn width_hz(&self) -> f64 {
        self.spectral_linspace.width_hz()
    }

    pub fn width_ppm(&self) -> f64 {
        self.spectral_linspace.width_ppm()
    }

    pub fn center_hz(&self) -> f64 {
        self.spectral_linspace.center_hz()
    }

    pub fn center_ppm(&self) -> f64 {
        self.spectral_linspace.center_ppm()
    }

    pub fn step_hz(&self) -> f64 {
        self.spectral_linspace.step_hz()
    }

    pub fn step_ppm(&self) -> f64 {
        self.spectral_linspace.step_ppm()
    }

    pub fn frequencies(&self) -> impl Iterator<Item = f64> {
        self.spectral_linspace.frequencies()
    }

    pub fn chemical_shifts(&self) -> impl Iterator<Item = f64> {
        self.spectral_linspace.chemical_shifts()
    }

    pub fn intensities(&self) -> &[f64] {
        &self.intensities
    }

    pub fn signal_boundaries_indices(&self) -> (usize, usize) {
        self.signal_boundaries
    }

    pub fn signal_boundaries_frequencies(&self) -> SignalBoundaries {
        // unwrapping is safe because signal_boundaries is validated during construction
        SignalBoundaries::Frequencies(
            self.spectral_linspace
                .index_to_hz(self.signal_boundaries.0)
                .unwrap(),
            self.spectral_linspace
                .index_to_hz(self.signal_boundaries.1)
                .unwrap(),
        )
    }

    pub fn signal_boundaries_ppm(&self) -> SignalBoundaries {
        // unwrapping is safe because signal_boundaries is validated during construction
        SignalBoundaries::ChemicalShifts(
            self.spectral_linspace
                .index_to_ppm(self.signal_boundaries.0)
                .unwrap(),
            self.spectral_linspace
                .index_to_ppm(self.signal_boundaries.1)
                .unwrap(),
        )
    }

    pub fn signal_boundaries_relative(&self) -> SignalBoundaries {
        // unwrapping is safe because signal_boundaries is validated during construction
        SignalBoundaries::Relative(
            self.spectral_linspace
                .index_to_relative(self.signal_boundaries.0)
                .unwrap(),
            self.spectral_linspace
                .index_to_relative(self.signal_boundaries.1)
                .unwrap(),
        )
    }

    pub fn set_id<T: Into<String>>(&mut self, id: T) {
        self.id = Some(id.into());
    }

    pub fn clear_id(&mut self) {
        self.id = None;
    }

    pub fn set_nucleus<T: Into<Nucleus>>(&mut self, nucleus: T) {
        self.nucleus = Some(nucleus.into());
    }

    pub fn clear_nucleus(&mut self) {
        self.nucleus = None;
    }

    pub fn set_frequency_range(&mut self, frequency_range: (f64, f64)) -> Result<()> {
        self.spectral_linspace
            .set_frequency_range(frequency_range)
    }

    pub fn set_spectrometer_frequency(&mut self, spectrometer_frequency: f64) -> Result<()> {
        self.spectral_linspace
            .set_spectrometer_frequency(spectrometer_frequency)
    }

    pub fn set_shift_reference<T: Into<ShiftReference>>(&mut self, reference: T) -> Result<()> {
        self.spectral_linspace
            .set_shift_reference(reference)
    }

    pub fn set_signal_boundaries(&mut self, signal_boundaries: SignalBoundaries) -> Result<()> {
        self.signal_boundaries =
            Self::validate_boundaries(signal_boundaries, &self.spectral_linspace)?;

        Ok(())
    }

    fn validate_intensities<I>(intensities: I) -> Result<Arc<[f64]>>
    where
        I: IntoIterator<Item = f64>,
    {
        let mut invalid_positions = Vec::<usize>::new();
        let intensities = intensities
            .into_iter()
            .enumerate()
            .map(|(position, intensity)| {
                if !intensity.is_finite() {
                    invalid_positions.push(position);
                }

                intensity
            })
            .collect::<Arc<[f64]>>();

        if intensities.is_empty() {
            return Err(Error::empty_data());
        }
        if !invalid_positions.is_empty() {
            return Err(Error::invalid_intensities(
                intensities.as_ref().to_vec(),
                invalid_positions,
            ));
        }

        Ok(intensities)
    }

    fn validate_boundaries(
        signal_boundaries: SignalBoundaries,
        linspace: &SpectralLinspace,
    ) -> Result<(usize, usize)> {
        match signal_boundaries {
            SignalBoundaries::Relative(start, end) => {
                match start.is_finite()
                    && end.is_finite()
                    && 0.0 <= start
                    && start <= 1.0
                    && 0.0 <= end
                    && end <= 1.0
                {
                    true => Ok((
                        linspace.relative_to_fractional(start).floor() as usize,
                        linspace.relative_to_fractional(end).ceil() as usize,
                    )),
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        (0.0, 1.0),
                    )),
                }
            }
            SignalBoundaries::Frequencies(start, end) => {
                match linspace.contains_hz(start) && linspace.contains_hz(end) {
                    true => {
                        let start = linspace.hz_to_fractional(start);
                        let end = linspace.hz_to_fractional(end);

                        match start < end {
                            true => Ok((start.floor() as usize, end.ceil() as usize)),
                            false => Ok((end.floor() as usize, start.ceil() as usize)),
                        }
                    }
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        linspace.range_ppm(),
                    )),
                }
            }
            SignalBoundaries::ChemicalShifts(start, end) => {
                match linspace.contains_ppm(start) && linspace.contains_ppm(end) {
                    true => {
                        let start = linspace.ppm_to_fractional(start);
                        let end = linspace.ppm_to_fractional(end);

                        match start < end {
                            true => Ok((start.floor() as usize, end.ceil() as usize)),
                            false => Ok((end.floor() as usize, start.ceil() as usize)),
                        }
                    }
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        linspace.range_ppm(),
                    )),
                }
            }
        }
    }
}
