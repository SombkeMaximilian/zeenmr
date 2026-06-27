//! Deconvolution error types.

use std::sync::Arc;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out the full type name directly, and is
/// broadly used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during the deconvolution process.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// The source of the error, if any.
    source: Option<Arc<Self>>,
}

/// The kind of `Error` that can occur during the deconvolution process.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Kind {
    /// Received a non-finite float value.
    ///
    /// Since the data structures provided by this library are intended to be
    /// used in numerical computation contexts, non-finite float values would
    /// corrupt all further processing steps and are therefore not allowed at
    /// the user boundary.
    NonFiniteFloat,
    /// No peaks were detected in the input data.
    ///
    /// Indicates that intensities may have been read incorrectly or something
    /// went wrong during the smoothing process.
    NoPeaksDetected,
    /// No peaks were found in the part of the spectrum where signals would be
    /// expected.
    ///
    /// This may happen if there are no signals in the signal region, either
    /// due to incorrectly setting its boundaries or if all the signals were
    /// filtered out by the peak finding algorithm.
    EmptySignalRegion,
    /// No peaks were found in the part of the spectrum where random signals due
    /// to noise would be expected.
    ///
    /// Some peak finding algorithms use the estimated noise from regions of a
    /// spectrum where no signals are expected filter out peaks in the signal
    /// region. If there is no noise, i.e., no peaks can be found, this process
    /// might silently fail. This error avoids that issue.
    EmptySignalFreeRegion,
    /// Ignore region is invalid.
    ///
    /// Both bounds of a region to ignore must be finite floats.
    InvalidIgnoreRegion,
}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self { kind, source: None }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::NonFiniteFloat => "non-finite float value received",
            Kind::NoPeaksDetected => "no peaks detected",
            Kind::EmptySignalRegion => "no peaks found in signal region",
            Kind::EmptySignalFreeRegion => "no peaks found in signal-free region",
            Kind::InvalidIgnoreRegion => match &self.source {
                Some(_) => "ignore region contains non-finite values",
                None => unreachable!("valid ignore region falsely detected as invalid"),
            },
        };

        write!(f, "{description}")
    }
}

impl Error {
    /// Creates a [`NonFiniteFloat`] error.
    ///
    /// [`NonFiniteFloat`]: Kind::NonFiniteFloat
    pub(crate) fn non_finite_float() -> Self {
        Kind::NonFiniteFloat.into()
    }

    /// Creates a [`NoPeaksDetected`] error.
    ///
    /// [`NoPeaksDetected`]: Kind::NoPeaksDetected
    pub(crate) fn no_peaks_detected() -> Self {
        Kind::NoPeaksDetected.into()
    }

    /// Creates an [`EmptySignalRegion`] error.
    ///
    /// [`EmptySignalRegion`]: Kind::EmptySignalRegion
    pub(crate) fn empty_signal_region() -> Self {
        Kind::EmptySignalRegion.into()
    }

    /// Creates an [`EmptySignalFreeRegion`] error.
    ///
    /// [`EmptySignalFreeRegion`]: Kind::EmptySignalFreeRegion
    pub(crate) fn empty_signal_free_region() -> Self {
        Kind::EmptySignalFreeRegion.into()
    }

    /// Creates an [`InvalidIgnoreRegion`] error.
    ///
    /// [`InvalidIgnoreRegion`]: Kind::InvalidIgnoreRegion
    pub(crate) fn invalid_ignore_region() -> Self {
        Self {
            kind: Kind::InvalidIgnoreRegion,
            source: Some(Arc::new(Self::non_finite_float())),
        }
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// Returns the source of the error, if any.
    pub fn source(&self) -> Option<&Error> {
        self.source.as_deref()
    }
}
