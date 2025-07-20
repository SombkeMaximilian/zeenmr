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
    EmptyData {
        /// Number of chemical shift values in the spectrum.
        chemical_shifts: usize,
        /// Number of intensity values in the spectrum.
        intensities: usize,
    },
    /// The input data lengths are mismatched.
    ///
    /// The length of a [`Spectrum`] is not intended to be changed after it is
    /// constructed. A mismatch in the number of the chemical shift and
    /// intensity values would create an inconsistent [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::Spectrum
    DataLengthMismatch {
        /// Number of chemical shift values in the spectrum.
        chemical_shifts: usize,
        /// Number of intensity values in the spectrum.
        intensities: usize,
    },
    /// The chemical shifts are not uniformly spaced.
    ///
    /// The step size between two consecutive chemical shift values needs to be
    /// approximately consistent throughout the entire [`Spectrum`]. A situation
    /// where this is not the case can arise due to
    ///
    /// - An inconsistent step size between two values
    /// - The difference between two values being very close to zero
    /// - Non-finite values in the chemical shifts
    ///
    /// Note that this error will short-circuit the construction of the
    /// [`Spectrum`] at the first occurrence of non-uniform spacing, meaning
    /// later values are not checked.
    ///
    /// [`Spectrum`]: crate::Spectrum
    NonUniformSpacing {
        /// Position of the first chemical shift with non-uniform spacing.
        position: usize,
    },
    /// The intensities contain invalid values.
    ///
    /// Non-finite intensity values will lead to problems in further processing
    /// steps. Therefore, this state is considered inconsistent and results in
    /// an error.
    InvalidIntensities {
        /// Positions of the invalid intensities.
        positions: Vec<usize>,
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
            Kind::EmptyData {
                chemical_shifts,
                intensities,
            } => match (*chemical_shifts == 0, *intensities == 0) {
                (true, true) => "chemical shifts and intensities are empty".to_string(),
                (true, false) => "chemical shifts are empty".to_string(),
                (false, true) => "intensities are empty".to_string(),
                _ => unreachable!("chemical shifts/intensities falsely detected as empty"),
            },
            Kind::DataLengthMismatch {
                chemical_shifts,
                intensities,
            } => format!(
                "lengths of chemical shifts [{chemical_shifts}]\
                 and intensities [{intensities}] do not match"
            ),
            Kind::NonUniformSpacing { position } => {
                format!("non-uniform spacing of chemical shifts at position {position}")
            }
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
    pub(crate) fn empty_data(chemical_shifts: usize, intensities: usize) -> Self {
        Kind::EmptyData {
            chemical_shifts,
            intensities,
        }
        .into()
    }

    /// Creates a new [`DataLengthMismatch`] error.
    ///
    /// [`DataLengthMismatch`]: Kind::DataLengthMismatch
    pub(crate) fn data_length_mismatch(chemical_shifts: usize, intensities: usize) -> Self {
        Kind::DataLengthMismatch {
            chemical_shifts,
            intensities,
        }
        .into()
    }

    /// Creates a new [`NonUniformSpacing`] error.
    ///
    /// [`NonUniformSpacing`]: Kind::NonUniformSpacing
    pub(crate) fn non_uniform_spacing(position: usize) -> Self {
        Kind::NonUniformSpacing { position }.into()
    }

    /// Creates a new [`InvalidIntensities`] error.
    ///
    /// [`InvalidIntensities`]: Kind::InvalidIntensities
    pub(crate) fn invalid_intensities(positions: Vec<usize>) -> Self {
        Kind::InvalidIntensities { positions }.into()
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
