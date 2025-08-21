use crate::error::{Error, Result};
use crate::{Nucleus, ReferencingMethod, ShiftReference, SignalBoundaries, SpectralLinspace};
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data structure that represents a 1D NMR spectrum.
///
/// A `Spectrum` consists of an ordered, immutable set of intensity values along
/// a spectral axis defined by the recorded frequency range, the spectrometer
/// frequency, and a chemical shift reference. In addition to the measured data,
/// a `Spectrum` can store metadata such as the observed nucleus, an identifier,
/// and signal region boundaries, which define the portion of the spectrum
/// expected to contain real signals.
///
/// # Invariants
///
/// A valid `Spectrum` instance maintains the following conditions:
/// - None of the values are [`f64::NAN`], [`f64::INFINITY`], or
///   [`f64::NEG_INFINITY`].
/// - The frequency range contains only values greater than or equal to zero.
/// - The spectrometer frequency is a value greater than zero.
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
/// spectrum.set_signal_boundaries(SignalBoundaries::Relative(0.15, 0.85))?;
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
    /// Constructs a `Spectrum` from an iterator of `intensities` and parameters
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
    /// let spectrum = Spectrum::new(intensities, spectrometer_frequency, frequency_range)?;
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
        let signal_boundaries = (
            spectral_linspace
                .relative_to_fractional(0.1)
                .ceil() as usize,
            spectral_linspace
                .relative_to_fractional(0.9)
                .floor() as usize,
        );

        Ok(Self {
            id: None,
            nucleus: None,
            spectral_linspace,
            signal_boundaries,
            intensities,
        })
    }

    /// Returns the ID of the `Spectrum`, if available.
    ///
    /// At construction, the ID is `None`. Use [`set_id`] to set a custom
    /// identifier.
    ///
    /// [`set_id`]: Spectrum::set_id
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert!(spectrum.id().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Returns the nucleus observed in the NMR experiment, if available.
    ///
    /// At construction, the nucleus `None`. Use [`set_nucleus`] to set a custom
    /// nucleus.
    ///
    /// [`set_nucleus`]: Spectrum::set_nucleus
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert!(spectrum.nucleus().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn nucleus(&self) -> Option<&Nucleus> {
        self.nucleus.as_ref()
    }

    /// Returns the spectrometer frequency in MHz.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.spectrometer_frequency(), 600.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn spectrometer_frequency(&self) -> f64 {
        self.spectral_linspace.spectrometer_frequency()
    }

    /// Returns the number of data points in the `Spectrum`, sometimes
    /// informally referred to as its "resolution".
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_eq!(spectrum.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.intensities.len(), self.spectral_linspace.size());

        self.spectral_linspace.size()
    }

    /// Returns `true` if the `Spectrum` has no data points.
    ///
    /// A valid `Spectrum` is guaranteed to be non-empty, so this method will
    /// always return `false` and is only provided by convention.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert!(!spectrum.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        debug_assert_ne!(self.len(), 0);

        self.len() == 0
    }

    /// Returns the chemical shift reference of the `Spectrum`.
    ///
    /// At construction, a `ShiftReference` anchored at index 0 of the spectral
    /// axis is created, which corresponds to no shift. Use
    /// [`set_shift_reference`] to set a custom chemical shift reference.
    ///
    /// [`set_shift_reference`]: Spectrum::set_shift_reference
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.shift_reference().chemical_shift(), 0.0);
    /// assert_eq!(spectrum.shift_reference().index(), 0);
    /// assert!(spectrum.shift_reference().name().is_none());
    /// assert!(spectrum.shift_reference().method().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_reference(&self) -> &ShiftReference {
        self.spectral_linspace.shift_reference()
    }

    /// Returns the frequency range of the spectral axis in Hz.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.range_hz().0, 0.0);
    /// assert_approx_eq!(f64, spectrum.range_hz().1, 12000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn range_hz(&self) -> (f64, f64) {
        self.spectral_linspace.range_hz()
    }

    /// Returns the chemical shift range of the spectral axis in ppm.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.range_ppm().0, 0.0);
    /// assert_approx_eq!(f64, spectrum.range_ppm().1, 12000.0 / 600.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn range_ppm(&self) -> (f64, f64) {
        self.spectral_linspace.range_ppm()
    }

    /// Returns the width of the spectral axis in Hz.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.width_hz(), 12000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn width_hz(&self) -> f64 {
        self.spectral_linspace.width_hz()
    }

    /// Returns the width of the spectral axis in ppm.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.width_ppm(), 12000.0 / 600.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn width_ppm(&self) -> f64 {
        self.spectral_linspace.width_ppm()
    }

    /// Returns the center frequency of the spectral axis in Hz.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.center_hz(), 6000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn center_hz(&self) -> f64 {
        self.spectral_linspace.center_hz()
    }

    /// Returns the center chemical shift of the spectral axis in ppm.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.center_ppm(), 6000.0 / 600.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn center_ppm(&self) -> f64 {
        self.spectral_linspace.center_ppm()
    }

    /// Returns the step size of the spectral axis in Hz.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.step_hz(), 12000.0 / 2.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn step_hz(&self) -> f64 {
        self.spectral_linspace.step_hz()
    }

    /// Returns the step size of the spectral axis in ppm.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_approx_eq!(f64, spectrum.step_ppm(), 12000.0 / (600.0 * 2.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn step_ppm(&self) -> f64 {
        self.spectral_linspace.step_ppm()
    }

    /// Returns an iterator over the spectral frequencies in Hz.
    ///
    /// A new iterator is created each time this method is called, only
    /// computing the frequency values on demand.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let mut frequencies = spectrum.frequencies();
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(6000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(12000.0));
    /// assert!(frequencies.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn frequencies(&self) -> impl Iterator<Item = f64> + use<> {
        self.spectral_linspace.frequencies()
    }

    /// Returns an iterator over the chemical shifts in ppm.
    ///
    /// A new iterator is created each time this method is called, only
    /// computing the chemical shift values on demand.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let mut chemical_shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(6000.0 / 600.0));
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(12000.0 / 600.0));
    /// assert!(chemical_shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn chemical_shifts(&self) -> impl Iterator<Item = f64> + use<> {
        self.spectral_linspace.chemical_shifts()
    }

    /// Returns the signal intensities of the spectrum as a slice.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// assert_eq!(spectrum.intensities().len(), 3);
    /// assert_approx_eq!(f64, spectrum.intensities()[0], 1.0);
    /// assert_approx_eq!(f64, spectrum.intensities()[1], 2.0);
    /// assert_approx_eq!(f64, spectrum.intensities()[2], 3.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn intensities(&self) -> &[f64] {
        &self.intensities
    }

    /// Returns the indices of the signal boundaries of the `Spectrum`.
    ///
    /// The boundaries represent the minimal inclusive range of indices `[i, j]`
    /// that fully contains the signal region of the `Spectrum`. At
    /// construction, signal boundaries are initialized to the central 80%
    /// of the spectral axis, i.e., from the 10% to the 90% relative
    /// positions along the axis. See [`set_signal_boundaries`] to set
    /// custom boundaries.
    ///
    /// [`set_signal_boundaries`]: Spectrum::set_signal_boundaries
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new((0..=10).map(|i| i as f64), 600.0, (0.0, 12000.0))?;
    /// assert_eq!(spectrum.signal_boundaries(), (1, 9));
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal_boundaries(&self) -> (usize, usize) {
        self.signal_boundaries
    }

    /// Returns the signal boundaries of the `Spectrum` as frequencies in Hz.
    ///
    /// The boundaries represent the minimal inclusive range of frequencies `[a,
    /// b]` that fully contains the signal region of the `Spectrum`. At
    /// construction, signal boundaries are initialized to the central 80%
    /// of the spectral axis, i.e., from the 10% to the 90% relative
    /// positions along the axis range. See [`set_signal_boundaries`] to set
    /// custom boundaries. Note that provided signal boundaries are
    /// converted to the minimal inclusive range, so the values returned
    /// here may differ from the input.
    ///
    /// [`set_signal_boundaries`]: Spectrum::set_signal_boundaries
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new((0..=10).map(|i| i as f64), 600.0, (0.0, 12000.0))?;
    /// match spectrum.signal_boundaries_hz() {
    ///     SignalBoundaries::Frequencies(start, end) => {
    ///         assert_approx_eq!(f64, start, 1200.0);
    ///         assert_approx_eq!(f64, end, 10800.0);
    ///     }
    ///     _ => panic!("expected frequencies boundaries"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal_boundaries_hz(&self) -> SignalBoundaries {
        debug_assert!(self.signal_boundaries.0 < self.len());
        debug_assert!(self.signal_boundaries.1 < self.len());

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

    /// Returns the signal boundaries of the spectrum as chemical shifts in ppm.
    ///
    /// The boundaries represent the minimal inclusive range of chemical shifts
    /// `[a, b]` that fully contains the signal region of the `Spectrum`. At
    /// construction, signal boundaries are initialized to the central 90%
    /// of the spectral axis, i.e., from the 10% to the 90% relative
    /// positions along the axis range. See [`set_signal_boundaries`] to set
    /// custom boundaries. Note that provided signal boundaries are
    /// converted to the minimal inclusive range, so the values returned
    /// here may differ from the input.
    ///
    /// [`set_signal_boundaries`]: Spectrum::set_signal_boundaries
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new((0..=10).map(|i| i as f64), 600.0, (0.0, 12000.0))?;
    /// match spectrum.signal_boundaries_ppm() {
    ///     SignalBoundaries::ChemicalShifts(start, end) => {
    ///         assert_approx_eq!(f64, start, 1200.0 / 600.0);
    ///         assert_approx_eq!(f64, end, 10800.0 / 600.0);
    ///     }
    ///     _ => panic!("expected chemical shift boundaries"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal_boundaries_ppm(&self) -> SignalBoundaries {
        debug_assert!(self.signal_boundaries.0 < self.len());
        debug_assert!(self.signal_boundaries.1 < self.len());

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

    /// Returns the signal boundaries of the spectrum in relative units.
    ///
    /// The boundaries represent the minimal inclusive range of relative
    /// positions `[a, b]` that fully contains the signal region of the
    /// `Spectrum`. At construction, signal boundaries are initialized to the
    /// central 80% of the spectral axis, i.e., from the 10% to the 90%
    /// relative positions along the axis range. See
    /// [`set_signal_boundaries`] to set custom boundaries. Note that
    /// provided signal boundaries are converted to the minimal inclusive
    /// range, so the values returned here may differ from the input.
    ///
    /// [`set_signal_boundaries`]: Spectrum::set_signal_boundaries
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new((0..=10).map(|i| i as f64), 600.0, (0.0, 12000.0))?;
    /// match spectrum.signal_boundaries_relative() {
    ///     SignalBoundaries::Relative(start, end) => {
    ///         assert_approx_eq!(f64, start, 0.1);
    ///         assert_approx_eq!(f64, end, 0.9);
    ///     }
    ///     _ => panic!("expected relative boundaries"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal_boundaries_relative(&self) -> SignalBoundaries {
        debug_assert!(self.signal_boundaries.0 < self.len());
        debug_assert!(self.signal_boundaries.1 < self.len());

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

    /// Sets the ID of the `Spectrum`.
    ///
    /// At construction, the ID is `None`. This can be used to store a
    /// human-readable identifier for the spectrum, such as a file name or a
    /// sample name.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_id("example_spectrum");
    /// assert_eq!(spectrum.id(), Some("example_spectrum"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id<T: Into<String>>(&mut self, id: T) {
        self.id = Some(id.into());
    }

    /// Clears the ID of the `Spectrum`, setting it to `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_id("example_spectrum");
    /// assert_eq!(spectrum.id(), Some("example_spectrum"));
    /// spectrum.clear_id();
    /// assert!(spectrum.id().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_id(&mut self) {
        self.id = None;
    }

    /// Sets the nucleus observed in the NMR experiment.
    ///
    /// At construction, the observed nucleus is `None`. Supports conversion
    /// from string representation using the full English names as well as
    /// IUPAC atomic element notation (e.g., `1H`, `13C`). See [`Nucleus`]
    /// for more details on supported nuclei.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_nucleus("proton");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Hydrogen).as_ref());
    /// spectrum.set_nucleus("Carbon-13");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Carbon13).as_ref());
    /// spectrum.set_nucleus("    15n   ");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Nitrogen15).as_ref());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_nucleus<T: Into<Nucleus>>(&mut self, nucleus: T) {
        self.nucleus = Some(nucleus.into());
    }

    /// Clears the nucleus of the `Spectrum`, setting it to `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_nucleus("deuterium");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Deuterium).as_ref());
    /// spectrum.clear_nucleus();
    /// assert!(spectrum.nucleus().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_nucleus(&mut self) {
        self.nucleus = None;
    }

    /// Sets the frequency range of the spectral axis in Hz.
    ///
    /// Note that this does not adjust the chemical shift reference, which
    /// may lead to unexpected results.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency range contains non-finite or negative
    /// values.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_frequency_range((10000.0, 0.0))?;
    ///
    /// let mut frequencies = spectrum.frequencies();
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(10000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(5000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(0.0));
    /// assert!(frequencies.next().is_none());
    ///
    /// // chemical shift reference maps the first frequency to 0 ppm
    /// let mut chemical_shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(-5000.0 / 600.0));
    /// assert_approx_eq!(Option<f64>, chemical_shifts.next(), Some(-10000.0 / 600.0));
    /// assert!(chemical_shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_frequency_range(&mut self, frequency_range: (f64, f64)) -> Result<()> {
        self.spectral_linspace
            .set_frequency_range(frequency_range)
    }

    /// Sets the spectrometer frequency in MHz.
    ///
    /// This value is a divisor when calculating the chemical shift values. As
    /// such, it is important that this value does not approach zero. For
    /// typical NMR experiments, this should not be an issue.
    ///
    /// # Errors
    ///
    /// Returns an error if the spectrometer frequency is not finite or
    /// if it is less than or equal to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// spectrum.set_spectrometer_frequency(450.0)?;
    ///
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 450.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 450.0));
    /// assert!(shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_spectrometer_frequency(&mut self, spectrometer_frequency: f64) -> Result<()> {
        self.spectral_linspace
            .set_spectrometer_frequency(spectrometer_frequency)
    }

    /// Sets the chemical shift reference of the `Spectrum`.
    ///
    /// At construction, a `ShiftReference` anchored at index 0 of the spectral
    /// axis is created, which corresponds to no shift. `ShiftReference` can be
    /// created from a tuple of a chemical shift value in ppm and an index
    /// in the spectral axis, or from just a chemical shift value (the index is
    /// 0 in this case). When using [`ShiftReference::new`], a name and a method
    /// can also be provided to describe the reference compound.
    ///
    /// # Errors
    ///
    /// Returns an error if the chemical shift value of the `ShiftReference`
    /// is not finite, or if the index is out of bounds of the spectral axis.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    ///
    /// // map the first frequency to -5.0 ppm
    /// spectrum.set_shift_reference(-5.0)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-5.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 5.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 5.0));
    ///
    /// // map the frequency at index 1 to 0.0 ppm
    /// spectrum.set_shift_reference((0.0, 1))?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    ///
    /// // create a custom shift reference with a name and method
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference<T: Into<ShiftReference>>(&mut self, reference: T) -> Result<()> {
        self.spectral_linspace
            .set_shift_reference(reference)
    }

    /// Sets the chemical shift reference value of the `Spectrum`.
    ///
    /// Useful for fine-tuning the chemical shift reference without changing all
    /// of its properties.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    ///
    /// // create a custom shift reference with a name and method
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    ///
    /// // adjust the chemical shift reference value from 0.0 to 1.0 ppm
    /// spectrum.set_shift_reference_value(1.0)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-9.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 9.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 9.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_value(&mut self, chemical_shift: f64) -> Result<()> {
        self.spectral_linspace
            .set_shift_reference_value(chemical_shift)?;

        Ok(())
    }

    /// Sets the chemical shift reference index of the `Spectrum`.
    ///
    /// Useful for fine-tuning the chemical shift reference without changing all
    /// of its properties.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    ///
    /// // create a custom shift reference with a name and method
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    ///
    /// // adjust the chemical shift reference index to the first point
    /// spectrum.set_shift_reference_index(0)?;
    /// let mut shifts = spectrum.chemical_shifts();
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_index(&mut self, index: usize) -> Result<()> {
        self.spectral_linspace
            .set_shift_reference_index(index)?;

        Ok(())
    }

    /// Sets the name of the chemical shift reference of the `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// spectrum.set_shift_reference_name("new ref name");
    /// assert_eq!(spectrum.shift_reference().name(), Some("new ref name"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_name<T: Into<String>>(&mut self, name: T) {
        self.spectral_linspace
            .set_shift_reference_name(name);
    }

    /// Clears the name of the chemical shift reference of the `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// spectrum.clear_shift_reference_name();
    /// assert!(spectrum.shift_reference().name().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_shift_reference_name(&mut self) {
        self.spectral_linspace
            .clear_shift_reference_name();
    }

    /// Sets the referencing method of the chemical shift reference of the
    /// `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// spectrum.set_shift_reference_method("external");
    /// assert_eq!(
    ///     spectrum.shift_reference().method(),
    ///     Some(ReferencingMethod::External).as_ref()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_method<T: Into<ReferencingMethod>>(&mut self, method: T) {
        self.spectral_linspace
            .set_shift_reference_method(method);
    }

    /// Clears the referencing method of the chemical shift reference of the
    /// `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(vec![1.0, 2.0, 3.0], 600.0, (0.0, 12000.0))?;
    /// let reference = ShiftReference::new(0.0, 1, Some("example ref"), Some("internal"));
    /// spectrum.set_shift_reference(reference)?;
    /// spectrum.clear_shift_reference_method();
    /// assert!(spectrum.shift_reference().method().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_shift_reference_method(&mut self) {
        self.spectral_linspace
            .clear_shift_reference_method();
    }

    /// Sets the signal boundaries of the `Spectrum`.
    ///
    /// Boundaries are internally converted to the minimal inclusive range
    /// `[i, j]` that fully contains all discrete points within the provided
    /// signal region. As such, the input of this method cannot be recovered.
    /// Consider the following coordinate pairs:
    ///
    /// | Index | Frequency  | Intensity |
    /// |-------|------------|-----------|
    /// | 0     | 0.0 Hz     | 1.0       |
    /// | 1     | 1200.0 Hz  | 2.0       |
    /// | 2     | 2400.0 Hz  | 3.0       |
    /// | 3     | 3600.0 Hz  | 4.0       |
    /// | 4     | 4800.0 Hz  | 5.0       |
    /// | 5     | 6000.0 Hz  | 6.0       |
    /// | 6     | 7200.0 Hz  | 5.0       |
    /// | 7     | 8400.0 Hz  | 4.0       |
    /// | 8     | 9600.0 Hz  | 3.0       |
    /// | 9     | 10800.0 Hz | 2.0       |
    /// | 10    | 12000.0 Hz | 1.0       |
    ///
    /// If the signal boundaries are the frequency values `2000 Hz` and
    /// `10000 Hz`, the values with indices `2` to `8` are fully contained
    /// within this range. The minimal inclusive range is therefore `[2, 8]`.
    /// When converted back to frequencies, the boundaries are `2400 Hz` and
    /// `9600 Hz`, i.e., the discrete data points closest to the provided
    /// boundaries from within.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided signal boundaries contain values that
    /// are not finite or not within bounds of the spectral axis.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{SignalBoundaries, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new((0..=10).map(|i| i as f64), 600.0, (0.0, 12000.0))?;
    /// spectrum.set_signal_boundaries(SignalBoundaries::Frequencies(2000.0, 10000.0))?;
    /// assert_eq!(spectrum.signal_boundaries(), (2, 8));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_signal_boundaries(&mut self, signal_boundaries: SignalBoundaries) -> Result<()> {
        self.signal_boundaries = self.validate_boundaries(signal_boundaries)?;

        Ok(())
    }

    /// Validates the intensities and returns them in the internal format.
    /// Returns an error if any of the values are not finite or if the iterator
    /// is empty.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`EmptyData`](crate::error::Kind::EmptyData)
    /// - [`InvalidIntensities`](crate::error::Kind::InvalidIntensities)
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

    /// Validates the signal boundaries and returns their indices in the
    /// spectral linspace. Returns an error if the boundaries are not valid
    /// according to the spectral linspace.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`InvalidSignalBoundaries`](crate::error::Kind::InvalidSignalBoundaries)
    fn validate_boundaries(&self, signal_boundaries: SignalBoundaries) -> Result<(usize, usize)> {
        match signal_boundaries {
            SignalBoundaries::Relative(start, end) => {
                match start.is_finite()
                    && end.is_finite()
                    && (0.0..=1.0).contains(&start)
                    && (0.0..=1.0).contains(&end)
                {
                    true => Ok((
                        self.spectral_linspace
                            .relative_to_fractional(start)
                            .ceil() as usize,
                        self.spectral_linspace
                            .relative_to_fractional(end)
                            .floor() as usize,
                    )),
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        (0.0, 1.0),
                    )),
                }
            }
            SignalBoundaries::Frequencies(start, end) => {
                match self.spectral_linspace.contains_hz(start)
                    && self.spectral_linspace.contains_hz(end)
                {
                    true => {
                        let start = self.spectral_linspace.hz_to_fractional(start);
                        let end = self.spectral_linspace.hz_to_fractional(end);

                        match start < end {
                            true => Ok((start.ceil() as usize, end.floor() as usize)),
                            false => Ok((end.ceil() as usize, start.floor() as usize)),
                        }
                    }
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        self.spectral_linspace.range_ppm(),
                    )),
                }
            }
            SignalBoundaries::ChemicalShifts(start, end) => {
                match self.spectral_linspace.contains_ppm(start)
                    && self.spectral_linspace.contains_ppm(end)
                {
                    true => {
                        let start = self.spectral_linspace.ppm_to_fractional(start);
                        let end = self.spectral_linspace.ppm_to_fractional(end);

                        match start < end {
                            true => Ok((start.ceil() as usize, end.floor() as usize)),
                            false => Ok((end.ceil() as usize, start.floor() as usize)),
                        }
                    }
                    false => Err(Error::invalid_signal_boundaries(
                        signal_boundaries,
                        self.spectral_linspace.range_ppm(),
                    )),
                }
            }
        }
    }
}
