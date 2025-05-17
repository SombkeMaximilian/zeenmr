use crate::{deconvolution, spectrum};

/// A specialized [`Result`] type for the Metabodecon library.
///
/// This type alias is used to avoid writing out `Result<T, metabodecon::Error>`
/// directly, and is broadly used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors encountered by the Metabodecon library.
///
/// Marked as non-exhaustive because some variants will only be available with
/// certain features enabled.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// An error that occurred during [`Spectrum`] construction or parsing.
    ///
    /// [`Spectrum`]: spectrum::Spectrum
    Spectrum(spectrum::error::Error),
    /// An error that occurred during the [`deconvolution`] process.
    Deconvolution(deconvolution::error::Error),
    /// Wrapper for errors from [`std::io`].
    IoError(std::io::Error),
}

impl std::error::Error for Error {}

impl From<spectrum::error::Error> for Error {
    fn from(value: spectrum::error::Error) -> Self {
        Error::Spectrum(value)
    }
}

impl From<deconvolution::error::Error> for Error {
    fn from(value: deconvolution::error::Error) -> Self {
        Error::Deconvolution(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::IoError(value)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self {
            Error::Spectrum(ref e) => e.fmt(f),
            Error::Deconvolution(ref e) => e.fmt(f),
            Error::IoError(ref e) => e.fmt(f),
        }
    }
}
