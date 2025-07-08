//! Error types for the alignment process.

use crate::alignment::{AlignmentStrategy, FilteringSettings, SolvingSettings};

/// An `Error` that occurred during the alignment process.
///
/// This type of error will occur when the [`Aligner`] was not configured
/// properly or when incompatible [`Deconvolution`]s are being aligned.
///
/// [`Aligner`]: crate::alignment::Aligner
/// [`Deconvolution`]: crate::deconvolution::Deconvolution
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Debug)]
pub struct Error {
    /// The `Kind` of error that occurred.
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

/// The kind of `Error` that can occur during the [`alignment`] process.
///
/// [`alignment`]: crate::alignment
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Kind {
    /// The provided alignment strategy is invalid.
    ///
    /// This can currently not occur, but will be used in the future when
    /// supporting more alignment strategies.
    InvalidAlignmentStrategy {
        /// The provided alignment strategy is invalid.
        strategy: AlignmentStrategy,
    },
    /// The provided filtering settings are invalid.
    ///
    /// Some configurations, such as a `max_distance` of 0 for a distance
    /// similarity filter, are invalid.
    InvalidFilteringSettings {
        /// The provided filtering settings.
        settings: FilteringSettings,
    },
    /// The provided solving settings are invalid.
    ///
    /// This can currently not occur, but will be used in the future when
    /// supporting more solving methods.
    InvalidSolvingSettings {
        /// The provided solving settings.
        settings: SolvingSettings,
    },
}

impl std::error::Error for Error {}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self { kind }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = match &self.kind {
            Kind::InvalidAlignmentStrategy { strategy } => match strategy {
                AlignmentStrategy::Reference(..) => {
                    unreachable!("reference alignment is always valid")
                }
                AlignmentStrategy::Pairwise => unreachable!("pairwise alignment is always valid"),
            },
            Kind::InvalidFilteringSettings { settings } => match settings {
                FilteringSettings::DistanceSimilarity {
                    max_distance,
                    min_similarity,
                    ..
                } => {
                    let finite = max_distance.is_finite() && min_similarity.is_finite();
                    let positive = *max_distance > 0.0 && *min_similarity > 0.0;
                    let max_similarity = *min_similarity < 1.0;

                    match (finite, positive, max_similarity) {
                        (false, _, _) => {
                            "max_distance and min_similarity must be finite".to_string()
                        }
                        (true, false, _) => {
                            "max_distance and min_similarity must be greater than 0".to_string()
                        }
                        (true, true, false) => "min_similarity must be less than 1".to_string(),
                        (true, true, true) => {
                            unreachable!("valid settings falsely detected as invalid")
                        }
                    }
                }
            },
            Kind::InvalidSolvingSettings { settings } => match settings {
                SolvingSettings::LinearProgramming => {
                    unreachable!("linear programming is always valid")
                }
            },
        };

        write!(f, "{}", description)
    }
}
