//! Error types for creating and validation intensity arrays.

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out the full type name directly, and is
/// broadly used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during construction or validation.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
}

/// The kind of `Error` that can occur during construction and validation.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Kind {
    /// Received a non-finite float.
    NonFiniteFloat,
    /// Received an index that is out of bounds of the array.
    OutOfBounds,
    /// A division by zero would have been attempted.
    ///
    /// For the purpose of numerical computations, attempting to divide by a
    /// value whose magnitude is less than `100 × epsilon` is considered
    /// dividing by zero.
    DivideByZero,
    /// An empty array was received.
    EmptyArray,
    /// An incorrect intensity kind was received.
    ///
    /// For example, an algorithm may make the assumption that its input is a
    /// magnitude spectrum, meaning the intensities are the absolute value of
    /// the complex channels, `s = sqrt(r^2 + i^2)`.
    IncorrectIntensityKind,
    /// No signal range could be found.
    ///
    /// This is not necessarily a hard error, but may suggest that the wrong
    /// test statistic was used to detect signals.
    NoSignal,
}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self { kind }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind {
            Kind::NonFiniteFloat => "non-finite float value received",
            Kind::OutOfBounds => "out of bounds access",
            Kind::DivideByZero => "attempted division by zero",
            Kind::EmptyArray => "array must have at least one element",
            Kind::IncorrectIntensityKind => "incorrect intensity kind",
            Kind::NoSignal => "no signal detected",
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

    /// Creates a [`OutOfBounds`] error.
    ///
    /// [`OutOfBounds`]: Kind::OutOfBounds
    pub(crate) fn out_of_bounds() -> Self {
        Kind::OutOfBounds.into()
    }

    /// Creates a [`DivisionByZero`] error.
    ///
    /// [`DivisionByZero`]: Kind::DivideByZero
    pub(crate) fn divide_by_zero() -> Self {
        Kind::DivideByZero.into()
    }

    /// Creates an [`EmptyArray`] error.
    ///
    /// [`EmptyArray`]: Kind::EmptyArray
    pub(crate) fn empty_array() -> Self {
        Kind::EmptyArray.into()
    }

    /// Creates a [`IncorrectIntensityKind`] error.
    ///
    /// [`IncorrectIntensityKind`]: Kind::IncorrectIntensityKind
    pub(crate) fn incorrect_intensity() -> Self {
        Kind::IncorrectIntensityKind.into()
    }

    /// Creates a [`NoSignal`] error.
    ///
    /// [`NoSignal`]: Kind::NoSignal
    pub(crate) fn no_signal() -> Self {
        Kind::NoSignal.into()
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> Kind {
        self.kind
    }
}
