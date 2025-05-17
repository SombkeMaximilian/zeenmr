//! Error types for the spectrum module.

use std::path::PathBuf;

/// The `Error` type for constructing a [`Spectrum`] or parsing 1D NMR data from
/// files.
///
/// [`Spectrum`]: crate::spectrum::Spectrum
///
/// This type of error is generally unrecoverable and indicates a problem with
/// the input data itself or the file format it is stored in. For example, the
/// input data is empty, a file of the Bruker TopSpin format is missing, or
/// metadata within one of the files is missing.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Debug)]
pub struct Error {
    /// `Kind` of the error that occurred.
    kind: Kind,
}

impl Error {
    /// Constructs a new `Error` from the given `Kind`.
    pub fn new(kind: Kind) -> Self {
        kind.into()
    }

    /// Returns the `Kind` of the `Error`.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}

/// The kind of `Error` that can occur while constructing a [`Spectrum`] or
/// reading 1D NMR data.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
///
/// [`Spectrum`]: crate::spectrum::Spectrum
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Kind {
    /// The input data is empty.
    ///
    /// The length of a [`Spectrum`] is not intended to be changed after it is
    /// constructed, so an empty [`Spectrum`] is simply not useful.
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    EmptyData {
        /// Number of elements in the chemical shifts vector.
        chemical_shifts: usize,
        /// Number of elements in the intensities vector.
        intensities: usize,
    },
    /// The input data lengths are mismatched.
    ///
    /// The length of a [`Spectrum`] is not intended to be changed after it is
    /// constructed. A mismatch in the lengths of the chemical shifts and
    /// intensities vectors would create an inconsistent [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    DataLengthMismatch {
        /// Number of elements in the chemical shifts vector.
        chemical_shifts: usize,
        /// Number of elements in the intensities vector.
        intensities: usize,
    },
    /// The chemical shifts are not uniformly spaced.
    ///
    /// The step size between two consecutive chemical shift values needs to be
    /// consistent throughout the entire [`Spectrum`]. A situation where this is
    /// not the case can arise due to
    /// - an inconsistent step size between two values
    /// - the difference between two values being very close to zero
    /// - non-finite values in the chemical shifts
    ///
    /// Note that this error will short-circuit the construction of the
    /// [`Spectrum`] at the first occurrence of non-uniform spacing, meaning
    /// subsequent values are not checked.
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    NonUniformSpacing {
        /// Step size computed from the first two chemical shifts.
        step_size: f64,
        /// Positions of the chemical shifts that are not uniformly spaced.
        positions: (usize, usize),
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
    /// found is in the center of the [`Spectrum`], with signal free regions on
    /// either side. The following conditions are checked:
    /// - Signal boundaries are finite values
    /// - Signal boundaries are within the range of the chemical shifts
    /// - Signal region width is not close to zero
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    InvalidSignalBoundaries {
        /// Signal boundaries of the spectrum.
        signal_boundaries: (f64, f64),
        /// Range of the chemical shifts.
        chemical_shifts_range: (f64, f64),
    },

    /// Metadata is missing from a file of the various formats.
    ///
    /// This indicates that the stored data was corrupted or that the format of
    /// the file is not as expected. If you have a dataset that you believe
    /// should be parsable but is not, open an [issue] and provide the dataset.
    ///
    /// [issue]: https://github.com/SombkeMaximilian/metabodecon-rust/issues
    MissingMetadata {
        /// Path to the file where the metadata was expected.
        path: PathBuf,
        /// Key that was used to search for the metadata.
        key: String,
    },
    /// Metadata is malformed in a file of the various formats.
    ///
    /// This indicates that the stored data was corrupted or that the format of
    /// the file is not as expected. If you have a dataset that you believe
    /// should be parsable but is not, open an [issue] and provide the dataset.
    ///
    /// [issue]: https://github.com/SombkeMaximilian/metabodecon-rust/issues
    MalformedMetadata {
        /// Path to the file where the metadata was expected.
        path: PathBuf,
        /// Key that was used to search for the metadata.
        key: String,
        /// Additional details about the malformed metadata.
        details: String,
    },
    /// The data block in a JCAMP-DX file is missing.
    MissingData {
        /// Path to the file where the data block was expected.
        path: PathBuf,
    },
    /// The data block in a JCAMP-DX file is malformed.
    MalformedData {
        /// Path to the file where the data block was expected.
        path: PathBuf,
        /// Additional details about the malformed data block.
        details: String,
    },
    /// The JCAMP-DX file is not supported.
    ///
    /// The JCAMP-DX format has different versions, and not all of them can be
    /// parsed at this time. Currently supported is:
    /// - Version 5.xx and 6.xx
    /// - NMR SPECTRUM data type
    /// - XYDATA and NTUPLES data classes
    ///
    /// If you have a file that you believe should be parsable but is not, open
    /// an [issue] and provide the file.
    ///
    /// [issue]: https://github.com/SombkeMaximilian/metabodecon-rust/issues
    UnsupportedJcampDxFile,
}

impl std::error::Error for Error {}

impl From<Kind> for Error {
    fn from(value: Kind) -> Self {
        Self { kind: value }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let description = match &self.kind {
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
                "lengths of chemical shifts [{}] and intensities [{}] do not match",
                chemical_shifts, intensities
            ),
            Kind::NonUniformSpacing {
                step_size,
                positions,
            } => match (step_size.is_finite(), *step_size > crate::CHECK_PRECISION) {
                (false, _) => format!(
                    "step size of the chemical shifts could not be computed \
                     due to non-finite values at indices [{}, {}]",
                    positions.0, positions.1
                ),
                (true, false) => format!(
                    "step size of the chemical shifts could not be computed \
                     due to (almost) identical values at indices [{}, {}]",
                    positions.0, positions.1
                ),
                (true, true) => format!(
                    "chemical shifts that are not uniformly spaced found at indices [{}, {}] \
                     (expected step size is [{}] from the first two values)",
                    positions.0, positions.1, step_size
                ),
            },
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
                let boundary_width = signal_boundaries.0 - signal_boundaries.1;
                let is_finite = signal_boundaries.0.is_finite() && signal_boundaries.1.is_finite();
                let is_spaced = f64::abs(boundary_width) > crate::CHECK_PRECISION;
                let is_contained = (signal_boundaries.0 > chemical_shifts_range.0
                    && signal_boundaries.1 < chemical_shifts_range.1)
                    || (signal_boundaries.0 < chemical_shifts_range.0
                        && signal_boundaries.1 > chemical_shifts_range.1);

                match (is_finite, is_spaced, is_contained) {
                    (false, _, _) => format!(
                        "signal boundaries [{}, {}] contain non-finite values",
                        signal_boundaries.0, signal_boundaries.1
                    ),
                    (true, false, true) => format!(
                        "signal boundaries [{}, {}] are \
                         (almost) equal",
                        signal_boundaries.0, signal_boundaries.1
                    ),
                    (true, true, false) => format!(
                        "signal boundaries [{}, {}] are \
                         not within the range of chemical shifts [{}, {}]",
                        signal_boundaries.0,
                        signal_boundaries.1,
                        chemical_shifts_range.0,
                        chemical_shifts_range.1
                    ),
                    (true, false, false) => format!(
                        "signal boundaries [{}, {}] are \
                         (almost) equal and \
                         not within the range of chemical shifts [{}, {}]",
                        signal_boundaries.0,
                        signal_boundaries.1,
                        chemical_shifts_range.0,
                        chemical_shifts_range.1
                    ),
                    _ => unreachable!("valid signal boundaries falsely detected as invalid"),
                }
            }
            Kind::MissingMetadata { path, key } => format!(
                "missing metadata \
                 expected in file at {:?} \
                 with key {}",
                path, key
            ),
            Kind::MalformedMetadata { path, key, details } => format!(
                "malformed metadata \
                 in file at {:?} \
                 with key {} \
                 ({})",
                path, key, details
            ),
            Kind::MissingData { path } => format!(
                "missing data block \"
                 expected in file at {:?}",
                path
            ),
            Kind::MalformedData { path, details } => format!(
                "malformed data block \
                 in file at {:?} \
                 ({})",
                path, details
            ),
            Kind::UnsupportedJcampDxFile => {
                "unsupported JCAMP-DX file (see documentation for supported file types)".to_string()
            }
        };

        write!(f, "{}", description)
    }
}
