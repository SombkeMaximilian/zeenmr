//! Spectrum error types.

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
    /// The frequency range of a [`Spectrum`] is expected to be a finite value.
    InvalidFrequencyRange {
        /// Frequency range of the spectrum.
        frequency_range: (f64, f64),
    },
    /// The spectrometer frequency is invalid.
    ///
    /// The spectrometer frequency of a [`Spectrum`] is expected to be a
    /// finite value.
    InvalidSpectrometerFrequency {
        /// Spectrometer frequency of the spectrum.
        spectrometer_frequency: f64,
    },
    /// The signal boundaries are invalid.
    ///
    /// A certain structure is expected from a 1D NMR [`Spectrum`] with respect
    /// to the regions of interest. The region where signals are expected to be
    /// found in the center of the [`Spectrum`], with signal-free regions on
    /// either side. The following conditions are checked:
    /// - Signal boundaries are finite values
    /// - Signal boundaries are within the range of the chemical shifts
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidSignalBoundaries {
        /// Signal boundaries of the spectrum.
        signal_boundaries: (f64, f64),
        /// Range of the chemical shifts.
        chemical_shifts_range: (f64, f64),
    },
    /// The index of a reference shift is out of bounds.
    ///
    /// This error occurs when the index of a reference shift is greater than or
    /// equal to the length of the chemical shifts in a [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::Spectrum
    ReferenceIndexOutOfBounds {
        /// Index of the reference shift.
        index: usize,
        /// Length of the chemical shifts.
        length: usize,
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
            Kind::InvalidIntensities { positions } => match positions.len() {
                0 => unreachable!("error should not be created without invalid intensities"),
                1 => format!(
                    "intensities contains a non-finite value at index [{}]",
                    positions[0]
                ),
                2..=5 => format!(
                    "intensities contain non-finite values at indices [{}]",
                    positions
                        .iter()
                        .map(|pos| pos.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ => format!(
                    "intensities contain non-finite values at indices [{}, ...] \
                     ({} invalid values)",
                    positions[..5]
                        .iter()
                        .map(|pos| pos.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    positions.len()
                ),
            },
            Kind::InvalidFrequencyRange { frequency_range } => {
                format!(
                    "frequency range [{}, {}] contains non-finite values",
                    frequency_range.0, frequency_range.1
                )
            }
            Kind::InvalidSpectrometerFrequency {
                spectrometer_frequency,
            } => {
                format!(
                    "spectrometer frequency [{}] is non-finite",
                    spectrometer_frequency
                )
            }
            Kind::InvalidSignalBoundaries {
                signal_boundaries,
                chemical_shifts_range,
            } => {
                let is_finite = signal_boundaries.0.is_finite() && signal_boundaries.1.is_finite();
                let is_contained = (signal_boundaries.0 > chemical_shifts_range.0
                    && signal_boundaries.1 < chemical_shifts_range.1)
                    || (signal_boundaries.0 < chemical_shifts_range.0
                        && signal_boundaries.1 > chemical_shifts_range.1);

                match (is_finite, is_contained) {
                    (false, _) => format!(
                        "signal boundaries [{}, {}] contain non-finite values",
                        signal_boundaries.0, signal_boundaries.1
                    ),
                    (true, false) => format!(
                        "signal boundaries [{}, {}] are \
                         not within the range of chemical shifts [{}, {}]",
                        signal_boundaries.0,
                        signal_boundaries.1,
                        chemical_shifts_range.0,
                        chemical_shifts_range.1
                    ),
                    _ => unreachable!("valid signal boundaries falsely detected as invalid"),
                }
            }
            Kind::ReferenceIndexOutOfBounds { index, length } => {
                format!(
                    "reference index [{index}] is out of bounds for chemical shifts of length [{length}]"
                )
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
        signal_boundaries: (f64, f64),
        chemical_shifts_range: (f64, f64),
    ) -> Self {
        Kind::InvalidSignalBoundaries {
            signal_boundaries,
            chemical_shifts_range,
        }
        .into()
    }

    /// Creates a new [`ReferenceIndexOutOfBounds`] error.
    ///
    /// [`ReferenceIndexOutOfBounds`]: Kind::ReferenceIndexOutOfBounds
    pub(crate) fn reference_index_out_of_bounds(index: usize, length: usize) -> Self {
        Kind::ReferenceIndexOutOfBounds { index, length }.into()
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}
