//! Bruker parsing error types.

use crate::jcampdx;
use std::io;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out the full type name directly, and is
/// broadly used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred while parsing a dataset.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// `Kind` of error that occurred.
    kind: Kind,
}

/// The kind of `Error` that can occur while parsing a dataset.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    /// A needed parameter is missing from `acqus` or `procs`.
    #[default]
    MissingParameter,
    /// An unexpected value was encountered for acquisition mode.
    ///
    /// Bruker stores the acquisition mode as integers (0, 1, 2, 3) under the
    /// key `AQ_mod`. Other values carry no meaning.
    UnknownAcquisitionMode,
    /// General I/O errors while reading files.
    IoError(io::ErrorKind),
    /// An error occurred while parsing the parameter files.
    ///
    /// See [`crate::error::Error`] for how to display these nicely.
    JcampDx(jcampdx::error::Error),
}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self { kind }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self {
            kind: Kind::IoError(value.kind()),
        }
    }
}

impl From<jcampdx::error::Error> for Error {
    fn from(value: jcampdx::error::Error) -> Self {
        Self {
            kind: Kind::JcampDx(value),
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = match self.kind {
            Kind::MissingParameter => "missing acqus/procs parameter",
            Kind::UnknownAcquisitionMode => "unknown acquisition mode",
            Kind::IoError(e) => return e.fmt(f),
            Kind::JcampDx(e) => return e.fmt(f),
        };

        write!(f, "{description}")
    }
}

impl Error {
    pub(crate) fn missing_parameter() -> Self {
        Kind::MissingParameter.into()
    }

    pub(crate) fn unknown_acquisition_mode() -> Self {
        Kind::UnknownAcquisitionMode.into()
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> Kind {
        self.kind
    }
}
