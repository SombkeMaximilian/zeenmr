//! Spectrum error types.

use crate::SignalBoundaries;

/// A specialized [`Result`] type.
///
/// This type alias avoids writing out `Result<T, zeenmr_spectrum::Error>`
/// directly, and is broadly used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// The `Error` type for constructing or manipulating a [`Spectrum`].
///
/// [`Spectrum`]: crate::Spectrum
///
/// This type of error is generally unrecoverable and indicates a problem with
/// the input data itself or the file format it is stored in. For example, the
/// input data is empty, a file of the Bruker TopSpin format is missing, or
/// metadata within one of the files is missing.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Debug)]
pub struct Error {
    /// `Kind` of error that occurred.
    kind: Kind,
}

/// The kind of `Error` that can occur while constructing or manipulating a
/// [`Spectrum`].
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
///
/// [`Spectrum`]: crate::Spectrum
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Kind {
    /// The input data is empty.
    ///
    /// The length of a [`Spectrum`] is not intended to be changed after it is
    /// constructed, so an empty [`Spectrum`] is simply not useful.
    ///
    /// [`Spectrum`]: crate::Spectrum
    EmptyData,
    /// The intensities contain invalid values.
    ///
    /// Non-finite intensity values will lead to problems in further processing
    /// steps. Therefore, this state is considered inconsistent and results in
    /// an error.
    InvalidIntensities {
        /// Positions of the invalid intensities.
        positions: Vec<usize>,
    },
    /// The frequency range is invalid.
    ///
    /// The frequency range of a [`Spectrum`] must be a tuple of finite values
    /// to generate chemical shifts from the frequency range.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidFrequencyRange {
        /// Frequency range of the spectrum.
        frequency_range: (f64, f64),
    },
    /// The spectrometer frequency is invalid.
    ///
    /// The spectrometer frequency of a [`Spectrum`] must be a finite value to
    /// generate chemical shifts from the frequency range.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidSpectrometerFrequency {
        /// Spectrometer frequency of the spectrum.
        spectrometer_frequency: f64,
    },
    /// The signal boundaries are invalid.
    ///
    /// When overriding the signal boundaries, which are normally automatically
    /// determined, some conditions must be met to ensure that the [`Spectrum`]
    /// remains consistent and usable:
    /// - Only finite values are allowed.
    /// - Values must be within the valid range.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidSignalBoundaries {
        /// Signal boundaries of the spectrum.
        signal_boundaries: SignalBoundaries,
        /// Range of the chemical shifts.
        valid_range: (f64, f64),
    },
    /// The index is out of bounds for the [`Spectrum`]'s size.
    ///
    /// This error occurs when a provided index is greater than or equal to the
    /// size of the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::Spectrum
    IndexOutOfBounds {
        /// Provided index that is out of bounds.
        index: usize,
        /// Size of the spectrum.
        size: usize,
    },
}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self { kind }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let description = match self.kind() {
            Kind::EmptyData => "intensities are empty".to_string(),
            Kind::InvalidIntensities { positions } => {
                let length = positions.len();
                let indices = positions[..usize::min(5, length)]
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");

                match length {
                    0 => unreachable!("error should not be created without invalid intensities"),
                    1 => format!("intensities contains a non-finite value at index [{indices}]"),
                    2..=5 => {
                        format!("intensities contain non-finite values at indices [{indices}]")
                    }
                    _ => format!(
                        "intensities contain non-finite values at indices [{indices}, ...] \
                         ({length} invalid values)",
                    ),
                }
            }
            Kind::InvalidFrequencyRange {
                frequency_range: (start, end),
            } => {
                format!("frequency range [{start}, {end}] contains non-finite values")
            }
            Kind::InvalidSpectrometerFrequency {
                spectrometer_frequency,
            } => format!("spectrometer frequency [{spectrometer_frequency}] is non-finite"),
            Kind::InvalidSignalBoundaries {
                signal_boundaries,
                valid_range,
            } => {
                let boundary_type = match signal_boundaries {
                    SignalBoundaries::Relative(_, _) => "relative units",
                    SignalBoundaries::Frequencies(_, _) => "frequencies",
                    SignalBoundaries::ChemicalShifts(_, _) => "chemical shifts",
                };
                let start = signal_boundaries.start();
                let end = signal_boundaries.end();
                let valid_start = valid_range.0;
                let valid_end = valid_range.1;
                let is_finite = start.is_finite() && end.is_finite();
                let is_contained = (start > valid_start && end < valid_end)
                    || (start < valid_start && end > valid_end);

                match (is_finite, is_contained) {
                    (false, _) => {
                        format!("signal boundaries [{start}, {end}] contain non-finite values",)
                    }
                    (true, false) => format!(
                        "signal boundaries [{start}, {end}] are not within \
                         the valid range of {boundary_type} [{valid_start}, {valid_end}]",
                    ),
                    _ => unreachable!("valid signal boundaries falsely detected as invalid"),
                }
            }
            Kind::IndexOutOfBounds { index, size } => {
                format!("index [{index}] is out of bounds for spectrum of size [{size}]")
            }
        };

        write!(f, "{description}")
    }
}

impl Error {
    /// Creates a new [`EmptyData`] error.
    ///
    /// [`EmptyData`]: Kind::EmptyData
    pub(crate) fn empty_data() -> Self {
        Kind::EmptyData.into()
    }

    /// Creates a new [`InvalidIntensities`] error.
    ///
    /// [`InvalidIntensities`]: Kind::InvalidIntensities
    pub(crate) fn invalid_intensities(positions: Vec<usize>) -> Self {
        Kind::InvalidIntensities { positions }.into()
    }

    /// Creates a new [`InvalidFrequencyRange`] error.
    ///
    /// [`InvalidFrequencyRange`]: Kind::InvalidFrequencyRange
    pub(crate) fn invalid_frequency_range(frequency_range: (f64, f64)) -> Self {
        Kind::InvalidFrequencyRange { frequency_range }.into()
    }

    /// Creates a new [`InvalidSpectrometerFrequency`] error.
    ///
    /// [`InvalidSpectrometerFrequency`]: Kind::InvalidSpectrometerFrequency
    pub(crate) fn invalid_spectrometer_frequency(spectrometer_frequency: f64) -> Self {
        Kind::InvalidSpectrometerFrequency {
            spectrometer_frequency,
        }
        .into()
    }

    /// Creates a new [`InvalidSignalBoundaries`] error.
    ///
    /// [`InvalidSignalBoundaries`]: Kind::InvalidSignalBoundaries
    pub(crate) fn invalid_signal_boundaries(
        signal_boundaries: SignalBoundaries,
        valid_range: (f64, f64),
    ) -> Self {
        Kind::InvalidSignalBoundaries {
            signal_boundaries,
            valid_range,
        }
        .into()
    }

    /// Creates a new [`IndexOutOfBounds`] error.
    ///
    /// [`IndexOutOfBounds`]: Kind::IndexOutOfBounds
    pub(crate) fn index_out_of_bounds(index: usize, size: usize) -> Self {
        Kind::IndexOutOfBounds { index, size }.into()
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}
