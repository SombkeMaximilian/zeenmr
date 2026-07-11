//! Deconvolution error types.

/// Sum error sum type with three variants.
///
/// Error types depend on the Deconvoluter components.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Error<E1, E2, E3> {
    kind: Kind<E1, E2, E3>,
}

/// Kind of the component errors.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Kind<E1, E2, E3> {
    /// An error occurred during smoothing.
    Smoothing(E1),
    /// An error occurred during peak finding.
    Finding(E2),
    /// An error occurred during fitting.
    Fitting(E3),
}

impl<E1, E2, E3> std::error::Error for Error<E1, E2, E3>
where
    E1: std::error::Error,
    E2: std::error::Error,
    E3: std::error::Error,
{
}

impl<E1, E2, E3> std::fmt::Display for Error<E1, E2, E3>
where
    E1: std::fmt::Display,
    E2: std::fmt::Display,
    E3: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            Kind::Smoothing(ref e) => write!(f, "smoothing: {e}"),
            Kind::Finding(ref e) => write!(f, "finding: {e}"),
            Kind::Fitting(ref e) => write!(f, "fitting: {e}"),
        }
    }
}

impl<E1, E2, E3> Error<E1, E2, E3> {
    /// Creates a smoothing error from the underlying error type.
    pub(crate) fn smoothing(error: E1) -> Self {
        Self {
            kind: Kind::Smoothing(error),
        }
    }

    /// Creates a finding error from the underlying error type.
    pub(crate) fn finding(error: E2) -> Self {
        Self {
            kind: Kind::Finding(error),
        }
    }

    /// Creates a fitting error from the underlying error type.
    pub(crate) fn fitting(error: E3) -> Self {
        Self {
            kind: Kind::Fitting(error),
        }
    }

    /// Returns the kind of error that occurred.
    pub fn kind(&self) -> &Kind<E1, E2, E3> {
        &self.kind
    }
}

/// Error two-sum type with two variants.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum EitherError<E1, E2> {
    /// First error occurred.
    First(E1),
    /// Second error occurred.
    Second(E2),
}

impl<E1, E2> std::error::Error for EitherError<E1, E2>
where
    E1: std::error::Error,
    E2: std::error::Error,
{
}

impl<E1, E2> std::fmt::Display for EitherError<E1, E2>
where
    E1: std::fmt::Display,
    E2: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::First(e1) => write!(f, "first: {e1}"),
            Self::Second(e2) => write!(f, "second: {e2}"),
        }
    }
}
