use crate::error::{Error, Result};
use crate::{
    ChemicalShiftRange, FrequencyRange, IndexRange, Nucleus, ReferencingMethod, ShiftReference,
    SpectralLinspace, SpectralRange, TryFromIndexRange, TryIntoIndexRange,
};
use std::sync::Arc;
use uom::si::f64::{Frequency, MagneticFluxDensity, Ratio};
use uom::typenum::P2;

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
/// - None of the values are `NAN`, `INF` or `-INF`.
/// - The larmor frequency's absolute value is not zero (<= [`f64::EPSILON`]).
/// - The index of the [`ShiftReference`] is within the bounds of the spectral
///   axis.
/// - Signal boundaries are within bounds of the spectral axis.
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
/// use num_traits::Zero;
/// use uom::si::f64::{Frequency, Ratio};
/// use uom::si::frequency::{hertz, megahertz};
/// use uom::si::ratio::part_per_million as ppm;
/// use zeenmr_spectrum::{ShiftReference, Spectrum};
///
/// # fn main() -> zeenmr_spectrum::error::Result<()> {
/// // Define larmor frequency and frequency range.
/// let larmor = Frequency::new::<megahertz>(600.0);
/// let range = (Frequency::zero(), Frequency::new::<hertz>(12000.0));
///
/// // Generate intensities using 3 Lorentzian peaks.
/// let intensities = (0..2_u32.pow(15))
///     .map(|i| i as f64 * range.1 / ((2_u32.pow(15) - 1) as f64))
///     .map(|f| (f / larmor).get::<ppm>())
///     .map(|x| {
///         // Reference signal centered at 5 ppm.
///         10.0 * 0.25 / (0.15_f64.powi(2) + (x - 5.0).powi(2))
///             // The left signal is centered at 8 ppm.
///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 8.0).powi(2))
///             // The right signal is centered at 12 ppm.
///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 12.0).powi(2))
///     });
///
/// // Create a Spectrum object.
/// let mut spectrum = Spectrum::new(intensities, larmor, range)?;
///
/// // Specify a chemical shift reference.
/// let reference = ShiftReference::new_with_meta(Ratio::zero(), 8192, "ref", "internal");
/// spectrum.set_shift_reference(reference)?;
///
/// // Add metadata.
/// spectrum.set_id("example_spectrum");
/// spectrum.set_nucleus("Deuterium");
/// spectrum.set_signal_boundaries((0.15, 0.85))?;
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
    pub(crate) linspace: SpectralLinspace,
    /// Boundaries of the signal region in the spectrum.
    signal_boundaries: IndexRange,
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
    /// Constructs a [`Spectrum`] from an iterator of `intensities` and
    /// parameters for the spectral axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the input data violates any of the invariants
    /// required for a valid [`Spectrum`]. The following conditions are checked:
    ///
    /// - Neither `intensities`, nor `larmor`, nor `range` contain non-finite
    ///   floats (no NAN, INF, -INF).
    /// - Absolute value of `larmor` is not zero (<= [`f64::EPSILON`]).
    /// - `intensities` are not empty.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let larmor = Frequency::new::<megahertz>(600.0);
    /// let range = (Frequency::zero(), Frequency::new::<hertz>(12000.0));
    /// let intensities = (0..2_u32.pow(15))
    ///     .map(|i| i as f64 * range.1 / ((2_u32.pow(15) - 1) as f64))
    ///     .map(|f| (f / larmor).get::<ppm>())
    ///     .map(|x| {
    ///         // Reference signal centered at 5 ppm.
    ///         10.0 * 0.25 / (0.15_f64.powi(2) + (x - 5.0).powi(2))
    ///             // The left signal is centered at 8 ppm.
    ///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 8.0).powi(2))
    ///             // The right signal is centered at 12 ppm.
    ///             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 12.0).powi(2))
    ///     });
    /// let spectrum = Spectrum::new(intensities, larmor, range)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<I, R>(intensities: I, larmor: Frequency, range: R) -> Result<Self>
    where
        R: Into<FrequencyRange>,
        I: IntoIterator<Item = f64>,
    {
        let intensities = Self::validate_intensities(intensities)?;
        let range = range.into();
        let reference = range.start / larmor;
        let spectral_linspace = SpectralLinspace::new(larmor, range, intensities.len(), reference)?;
        let signal_boundaries = (
            ((intensities.len() - 1) as f64 * 0.1).ceil() as usize,
            ((intensities.len() - 1) as f64 * 0.9).floor() as usize + 1,
        )
            .into();

        Ok(Self {
            id: None,
            nucleus: None,
            linspace: spectral_linspace,
            signal_boundaries,
            intensities,
        })
    }

    /// Returns the ID, if available.
    ///
    /// Use [`Spectrum::set_id`] to set a custom identifier.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert!(spectrum.id().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Returns the nucleus observed in the NMR experiment, if available.
    ///
    /// Use [`Spectrum::set_nucleus`] to set a custom nucleus.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert!(spectrum.nucleus().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn nucleus(&self) -> Option<&Nucleus> {
        self.nucleus.as_ref()
    }

    /// Returns the larmor frequency.
    ///
    /// This value is used to standardize the chemical shifts in the spectrum.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.larmor().get::<megahertz>(), 600.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn larmor(&self) -> Frequency {
        self.linspace.larmor()
    }

    /// Calculates the B_0 field strength from the larmor frequency and the
    /// observed nucleus, if available.
    ///
    /// The first value is the calculated B_0 field strength, the second value
    /// is the uncertainty based on the uncertainty of the gyromagnetic ratio.
    ///
    /// Uses the larmor equation:
    ///
    /// ```text
    /// ω₀ = -γ * B₀
    /// ```
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::magnetic_flux_density::tesla;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_nucleus("1H");
    /// let (b_field, uncertainty) = spectrum.b_field().expect("B0 missing");
    /// assert_approx_eq!(f64, b_field.get::<tesla>(), 14.091, epsilon = 1e-3);
    /// assert_approx_eq!(f64, uncertainty.get::<tesla>(), 5.957e-9, epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn b_field(&self) -> Option<(MagneticFluxDensity, MagneticFluxDensity)> {
        self.nucleus().and_then(|nucleus| {
            nucleus
                .gyromagnetic_ratio()
                .and_then(|(gyro, uncertainty)| {
                    Some((
                        self.larmor() / gyro,
                        self.larmor().abs() / gyro.powi(P2::new()) * uncertainty,
                    ))
                })
        })
    }

    /// Returns the number of data points in the `Spectrum`, sometimes
    /// informally referred to as its "resolution".
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_eq!(spectrum.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.intensities.len(), self.linspace.size());

        self.linspace.size()
    }

    /// Returns `true` if the [`Spectrum`] has no data points.
    ///
    /// A valid `Spectrum` is guaranteed to be non-empty, so this method will
    /// always return `false` and is only provided by convention.
    /// Deserialization may produce an empty [`Spectrum`].
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert!(!spectrum.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        debug_assert_ne!(self.len(), 0);

        self.len() == 0
    }

    /// Returns the chemical shift reference.
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
    /// use num_traits::Zero;
    /// use uom::si::f64::{Frequency, Ratio};
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.shift_reference().shift().get::<ppm>(), 0.0);
    /// assert_eq!(spectrum.shift_reference().index(), 0);
    /// assert!(spectrum.shift_reference().name().is_none());
    /// assert!(spectrum.shift_reference().method().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_reference(&self) -> &ShiftReference {
        self.linspace.shift_reference()
    }

    /// Returns the frequency range of the spectral axis.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.freq_range().start.get::<hertz>(), 0.0);
    /// assert_approx_eq!(f64, spectrum.freq_range().end.get::<hertz>(), 12000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn freq_range(&self) -> FrequencyRange {
        self.linspace.freq_range()
    }

    /// Returns the chemical shift range of the spectral axis.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.shift_range().start.get::<ppm>(), 0.0);
    /// assert_approx_eq!(f64, spectrum.shift_range().end.get::<ppm>(), 20.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_range(&self) -> ChemicalShiftRange {
        self.linspace.shift_range()
    }

    /// Returns the width of the spectral axis in terms of frequency.
    ///
    /// Always a positive value, even if the axis is in descending order.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.freq_width().get::<hertz>(), 12000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn freq_width(&self) -> Frequency {
        self.linspace.freq_width()
    }

    /// Returns the width of the spectral axis in terms of chemical shift.
    ///
    /// Always a positive value, even if the axis is in descending order.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.shift_width().get::<ppm>(), 20.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_width(&self) -> Ratio {
        self.linspace.width_ppm()
    }

    /// Returns the central frequency of the spectral axis.
    ///
    /// Not be one of the discrete data points if the [`Spectrum`]'s length is
    /// even.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.freq_center().get::<hertz>(), 6000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn freq_center(&self) -> Frequency {
        self.linspace.freq_center()
    }

    /// Returns the central chemical shift of the spectral axis.
    ///
    /// Not be one of the discrete data points if the [`Spectrum`]'s length is
    /// even.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.shift_center().get::<ppm>(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_center(&self) -> Ratio {
        self.linspace.shift_center()
    }

    /// Returns the step size of the spectral axis in terms of frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.freq_step().get::<hertz>(), 6000.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn freq_step(&self) -> Frequency {
        self.linspace.freq_step()
    }

    /// Returns the step size of the spectral axis in ppm.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// assert_approx_eq!(f64, spectrum.shift_step().get::<ppm>(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shift_step(&self) -> Ratio {
        self.linspace.shift_step()
    }

    /// Returns an iterator over the spectral frequencies.
    ///
    /// A new iterator is created each time this method is called, only
    /// computing the frequency values on demand.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// let mut frequencies = spectrum.frequencies().map(|f| f.get::<hertz>());
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(6000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(12000.0));
    /// assert!(frequencies.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn frequencies(&self) -> impl Iterator<Item = Frequency> + use<> {
        self.linspace.frequencies()
    }

    /// Returns an iterator over the chemical shifts.
    ///
    /// A new iterator is created each time this method is called, only
    /// computing the chemical shift values on demand.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0));
    /// assert!(shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn shifts(&self) -> impl Iterator<Item = Ratio> + use<> {
        self.linspace.shifts()
    }

    /// Returns the signal intensities of the spectrum as a slice.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
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

    /// Returns of the signal boundaries of the `Spectrum` as a range of type
    /// `R`.
    ///
    /// The boundaries represent the minimal range of indices `[i, j)` that
    /// fully contains the signal region of the `Spectrum`. At construction,
    /// signal boundaries are initialized to the central 80% of the spectral
    /// axis, i.e., from the 10% to the 90% relative positions along the axis.
    /// Use [`set_signal_boundaries`] to set custom boundaries.
    ///
    /// [`set_signal_boundaries`]: Spectrum::set_signal_boundaries
    ///
    /// # Possible Range Types
    ///
    /// The following types implement [`TryFromIndexRange`]:
    /// - [`IndexRange`]: A range of `usize` indices into the intensities,
    ///   expressed as `[i, j)`, where `i` is inclusive and `j` is exclusive.
    /// - [`FrequencyRange`]: A range of [`Frequency`] values, expressed as
    ///   `[f1, f2]`. Both bounds are inclusive and may appear in any order.
    /// - [`ChemicalShiftRange`]: A range of chemical shifts as [`Ratio`]
    ///   values, expressed as `[s1, s2]`. Both bounds are inclusive and may
    ///   appear in any order.
    /// - [`RelativeRange`]: A range of relative positions along the spectral
    ///   axis, expressed as `[r1, r2]`. Both bounds are inclusive, with values
    ///   in the normalized range `[0_f64, 1_f64]`.
    ///
    /// [`RelativeRange`]: crate::RelativeRange
    ///
    /// # Errors
    ///
    /// Infallible under normal circumstances, but it is possible to create
    /// invalid [`Spectrum`] instances through deserialization, in which case
    /// it returns an error if the deserialized boundaries are out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{
    ///     ChemicalShiftRange, FrequencyRange, IndexRange, RelativeRange, Spectrum,
    /// };
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    ///
    /// let freq_boundaries = spectrum.signal_boundaries::<FrequencyRange>()?;
    /// assert_approx_eq!(f64, freq_boundaries.start.get::<hertz>(), 1200.0);
    /// assert_approx_eq!(f64, freq_boundaries.end.get::<hertz>(), 10800.0);
    ///
    /// let shift_boundaries = spectrum.signal_boundaries::<ChemicalShiftRange>()?;
    /// assert_approx_eq!(f64, shift_boundaries.start.get::<ppm>(), 2.0);
    /// assert_approx_eq!(f64, shift_boundaries.end.get::<ppm>(), 18.0);
    ///
    /// let rel_boundaries = spectrum.signal_boundaries::<RelativeRange>()?;
    /// assert_approx_eq!(f64, rel_boundaries.start, 0.1);
    /// assert_approx_eq!(f64, rel_boundaries.end, 0.9);
    ///
    /// let index_boundaries = spectrum.signal_boundaries::<IndexRange>()?;
    /// assert_eq!(index_boundaries.start, 1);
    /// assert_eq!(index_boundaries.end, 10);
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal_boundaries<R>(&self) -> Result<R>
    where
        R: TryFromIndexRange,
    {
        debug_assert!(self.signal_boundaries.start < self.len());
        debug_assert!(self.signal_boundaries.end < self.len());

        R::try_from_index_range(self.signal_boundaries, self)
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
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_id("example_spectrum");
    /// assert_eq!(spectrum.id(), Some("example_spectrum"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id<T>(&mut self, id: T)
    where
        T: Into<String>,
    {
        self.id = Some(id.into());
    }

    /// Clears the ID of the `Spectrum`, setting it to `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
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
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_nucleus("proton");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Hydrogen).as_ref());
    /// spectrum.set_nucleus("Carbon-13");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Carbon13).as_ref());
    /// spectrum.set_nucleus("    15n   ");
    /// assert_eq!(spectrum.nucleus(), Some(Nucleus::Nitrogen15).as_ref());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_nucleus<T>(&mut self, nucleus: T)
    where
        T: Into<Nucleus>,
    {
        self.nucleus = Some(nucleus.into());
    }

    /// Clears the nucleus, setting it to `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::{Nucleus, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
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

    /// Sets the frequency range of the spectral axis.
    ///
    /// Note that this does not adjust the chemical shift reference, which
    /// may lead to unexpected results.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency range contains non-finite floats.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_range((
    ///     Frequency::new::<hertz>(10000.0),
    ///     Frequency::new::<hertz>(0.0),
    /// ))?;
    ///
    /// let mut frequencies = spectrum.frequencies().map(|f| f.get::<hertz>());
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(10000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(5000.0));
    /// assert_approx_eq!(Option<f64>, frequencies.next(), Some(0.0));
    /// assert!(frequencies.next().is_none());
    ///
    /// // chemical shift reference still maps the first frequency to 0 ppm
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-5000.0 / 600.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10000.0 / 600.0));
    /// assert!(shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_range(&mut self, range: (Frequency, Frequency)) -> Result<()> {
        self.linspace.set_range(range)
    }

    /// Sets the larmor frequency.
    ///
    /// This value is used to standardize the chemical shifts in the spectrum.
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    ///
    /// # Errors
    ///
    /// Returns an error if the larmor frequency is not finite or too small
    /// (<= [`f64::EPSILON`]).
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_larmor(Frequency::new::<megahertz>(450.0))?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(0.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 450.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 450.0));
    /// assert!(shifts.next().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_larmor(&mut self, larmor: Frequency) -> Result<()> {
        self.linspace.set_larmor(larmor)
    }

    /// Sets the larmor frequency based on the provided magnetic field
    /// strength and the currently set nucleus.
    ///
    /// Uses the larmor equation:
    ///
    /// ```text
    /// ω₀ = -γ * B₀
    /// ```
    ///
    /// See also: [NMRCentral](https://web.archive.org/web/20110926141002/http://nmrcentral.com/2011/08/chemical-shift/)
    ///
    /// # Errors
    ///
    /// Returns an error if nucleus is not set, or if it is [`Nucleus::Other`],
    /// or if `b_field` is not a finite value.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::{Frequency, MagneticFluxDensity};
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::magnetic_flux_density::tesla;
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_nucleus("1H");
    /// spectrum.set_larmor_by_b_field(MagneticFluxDensity::new::<tesla>(14.098))?;
    /// let larmor = spectrum.larmor().get::<megahertz>();
    /// assert_approx_eq!(f64, larmor, 600.257, epsilon = 1e-3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_larmor_by_b_field(&mut self, b_field: MagneticFluxDensity) -> Result<()> {
        let larmor = self.nucleus().and_then(|nucleus| {
            nucleus
                .gyromagnetic_ratio()
                .and_then(|(gyro, _)| Some(gyro * b_field))
        });

        match larmor {
            Some(freq) => self.set_larmor(freq),
            None => Err(Error::invalid_larmor(None)),
        }
    }

    /// Sets the chemical shift reference.
    ///
    /// At construction, a [`ShiftReference`] anchored at index 0 of the
    /// spectral axis is created, which corresponds to no shift. When using
    /// [`ShiftReference::new_with_meta`], a name and a method can also be
    /// provided to describe the reference compound. [`ShiftReference`] also
    /// implements the following traits for convenient construction:
    /// - [`From<Ratio>`]: first point in the [`Spectrum`], shift set to the
    ///   provided value.
    /// - [`From<usize>`]: provided point in the [`Spectrum`], shift set to 0.
    /// - [`From<(Ratio, usize)>`] and [`From<(usize, Ratio)>`]: provided shift
    ///   and index.
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
    /// use num_traits::Zero;
    /// use uom::si::f64::{Frequency, Ratio};
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    ///
    /// // map the first frequency to -5.0 ppm
    /// spectrum.set_shift_reference(Ratio::new::<ppm>(-5.0))?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-5.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 5.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 5.0));
    ///
    /// // map the frequency at index 1 to 0.0 ppm
    /// spectrum.set_shift_reference((Ratio::zero(), 1))?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    ///
    /// // create a custom shift reference with a name and method
    /// let reference = ShiftReference::new_with_meta(Ratio::zero(), 1, "example ref", "internal");
    /// spectrum.set_shift_reference(reference)?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference<T>(&mut self, reference: T) -> Result<()>
    where
        T: Into<ShiftReference>,
    {
        self.linspace.set_shift_reference(reference)
    }

    /// Sets the chemical shift reference value.
    ///
    /// Useful for fine-tuning the chemical shift reference without changing all
    /// of its properties.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::{Frequency, Ratio};
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_value(Ratio::new::<ppm>(1.0))?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(1.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 + 1.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 + 1.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_value(&mut self, shift: Ratio) -> Result<()> {
        self.linspace.set_shift_reference_value(shift)?;

        Ok(())
    }

    /// Sets the chemical shift reference index.
    ///
    /// Useful for fine-tuning the chemical shift reference without changing all
    /// of its properties.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ShiftReference, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_index(1)?;
    /// let mut shifts = spectrum.shifts().map(|s| s.get::<ppm>());
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(-10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(6000.0 / 600.0 - 10.0));
    /// assert_approx_eq!(Option<f64>, shifts.next(), Some(12000.0 / 600.0 - 10.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_index(&mut self, index: usize) -> Result<()> {
        self.linspace.set_shift_reference_index(index)?;

        Ok(())
    }

    /// Sets the name of the chemical shift reference of the `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_name("new ref name");
    /// assert_eq!(spectrum.shift_reference().name(), Some("new ref name"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_name<T: Into<String>>(&mut self, name: T) {
        self.linspace.set_shift_reference_name(name);
    }

    /// Clears the name of the chemical shift reference.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_name("example ref");
    /// spectrum.clear_shift_reference_name();
    /// assert!(spectrum.shift_reference().name().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_shift_reference_name(&mut self) {
        self.linspace.clear_shift_reference_name();
    }

    /// Sets the referencing method of the chemical shift reference.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::{ReferencingMethod, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_method("external");
    /// assert_eq!(
    ///     spectrum.shift_reference().method(),
    ///     Some(&ReferencingMethod::External)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_shift_reference_method<T: Into<ReferencingMethod>>(&mut self, method: T) {
        self.linspace.set_shift_reference_method(method);
    }

    /// Clears the referencing method of the chemical shift reference of the
    /// `Spectrum`.
    ///
    /// # Example
    ///
    /// ```
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::Spectrum;
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_shift_reference_method("external");
    /// spectrum.clear_shift_reference_method();
    /// assert!(spectrum.shift_reference().method().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_shift_reference_method(&mut self) {
        self.linspace.clear_shift_reference_method();
    }

    /// Sets the signal boundaries of the `Spectrum`.
    ///
    /// Boundaries are internally converted to the minimal range of indices
    /// `[i, j)` that fully contains all discrete points within the provided
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
    /// within this range. The minimal range is therefore `[2, 9)`. When
    /// converted back to frequencies, the boundaries are `2400 Hz` and
    /// `9600 Hz`, i.e., the discrete data points closest to the provided
    /// boundaries from within.
    ///
    /// # Possible Range Types
    ///
    /// The following types implement [`TryIntoIndexRange`]:
    /// - [`IndexRange`]: A range of `usize` indices into the intensities,
    ///   expressed as `[i, j)`, where `i` is inclusive and `j` is exclusive.
    /// - [`FrequencyRange`]: A range of [`Frequency`] values, expressed as
    ///   `[f1, f2]`. Both bounds are inclusive and may appear in any order.
    /// - [`ChemicalShiftRange`]: A range of chemical shifts as [`Ratio`]
    ///   values, expressed as `[s1, s2]`. Both bounds are inclusive and may
    ///   appear in any order.
    /// - [`RelativeRange`]: A range of relative positions along the spectral
    ///   axis, expressed as `[r1, r2]`. Both bounds are inclusive, with values
    ///   in the normalized range `[0_f64, 1_f64]`.
    ///
    /// [`RelativeRange`]: crate::RelativeRange
    ///
    /// These types each implement [`From<Range<T>>`] or
    /// [`From<RangeInclusive<T>>`] respectively, and [`From<(T, T)>`].
    ///
    /// [`From<(T, T)>`]: From
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
    /// use num_traits::Zero;
    /// use uom::si::f64::Frequency;
    /// use uom::si::frequency::{hertz, megahertz};
    /// use zeenmr_spectrum::{IndexRange, Spectrum};
    ///
    /// # fn main() -> zeenmr_spectrum::error::Result<()> {
    /// let mut spectrum = Spectrum::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ///     Frequency::new::<megahertz>(600.0),
    ///     (Frequency::zero(), Frequency::new::<hertz>(12000.0)),
    /// )?;
    /// spectrum.set_signal_boundaries((
    ///     Frequency::new::<hertz>(2000.0),
    ///     Frequency::new::<hertz>(10000.0),
    /// ))?;
    /// assert_eq!(spectrum.signal_boundaries::<IndexRange>()?.start, 2);
    /// assert_eq!(spectrum.signal_boundaries::<IndexRange>()?.end, 9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_signal_boundaries<T, R>(&mut self, signal_boundaries: R) -> Result<()>
    where
        R: Into<SpectralRange<T>>,
        SpectralRange<T>: TryIntoIndexRange,
    {
        let signal_boundaries = signal_boundaries.into();
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
            return Err(Error::invalid_intensities(invalid_positions));
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
    fn validate_boundaries<R>(&self, signal_boundaries: R) -> Result<IndexRange>
    where
        R: TryIntoIndexRange,
    {
        match signal_boundaries.try_into_index_range(self) {
            Ok(index_range) => Ok(index_range),
            Err(e) => Err(Error::invalid_signal_boundaries(e)),
        }
    }
}
