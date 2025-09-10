//! Spectrum error types.

use std::sync::Arc;

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
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Error {
    /// `Kind` of error that occurred.
    kind: Kind,
    /// Source of the error, if any.
    source: Option<Arc<Self>>,
}

/// The kind of `Error` that can occur while constructing or manipulating a
/// [`Spectrum`].
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
///
/// [`Spectrum`]: crate::Spectrum
#[non_exhaustive]
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Kind {
    /// Received a non-finite float value.
    ///
    /// Since the data structures provided by this library are intended to be
    /// used in numerical computation contexts, non-finite float values would
    /// corrupt all further processing steps and are therefore not allowed at
    /// the user boundary.
    ///
    /// [`InvalidShiftReference`]: Kind::InvalidShiftReference
    NonFiniteFloat,
    /// The input data is empty.
    ///
    /// The length of a [`Spectrum`] is not intended to be changed after it is
    /// constructed, so an empty [`Spectrum`] is simply not useful.
    ///
    /// [`Spectrum`]: crate::Spectrum
    EmptyData,
    /// Frequency range is invalid.
    ///
    /// The frequency range of a [`Spectrum`] must be a tuple of finite values
    /// to generate chemical shifts.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidRange,
    /// Larmor frequency is invalid.
    ///
    /// The larmor frequency of a [`Spectrum`] must be a finite, non-zero
    /// value to generate chemical shifts from the frequency range.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidLarmor,
    /// Signal boundaries are invalid.
    ///
    /// When overriding the signal boundaries, which are normally automatically
    /// determined, they must be finite floats and within the bounds of the
    /// spectral axis.
    ///
    /// [`Spectrum`]: crate::Spectrum
    InvalidSignalBoundaries,
    /// An index or a value is out of bounds.
    ///
    /// This error occurs when a provided index is greater than or equal to the
    /// size of the [`Spectrum`], or when a relative, frequency or chemical
    /// shift is not within the bounds of its spectral axis.
    ///
    /// [`Spectrum`]: crate::Spectrum
    OutOfBounds,
    /// Provided shift reference is invalid.
    ///
    /// This error can occur as a result of either a non-finite chemical shift
    /// or an out-of-bounds index.
    InvalidShiftReference,
    /// Received intensities containing non-finite floats.
    ///
    /// Since the data structures provided by this library are intended to be
    /// used in numerical computation contexts, non-finite float values would
    /// corrupt all further processing steps and are therefore not allowed at
    /// the user boundary.
    InvalidIntensities(Vec<usize>),
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

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let description = match self.kind() {
            Kind::NonFiniteFloat => "non-finite float value received".to_string(),
            Kind::EmptyData => "intensities are empty".to_string(),
            Kind::InvalidRange => "frequency range contains non-finite values".to_string(),
            Kind::InvalidLarmor => match &self.source {
                Some(_) => "larmor frequency is non-finite".to_string(),
                None => "larmor frequency is non-finite or near zero".to_string(),
            },
            Kind::InvalidSignalBoundaries => match &self.source {
                Some(source) => format!("invalid signal boundaries: {source}"),
                None => unreachable!("valid signal boundaries falsely detected as invalid"),
            },
            Kind::OutOfBounds => "index or value is out of bounds".to_string(),
            Kind::InvalidShiftReference => match &self.source {
                Some(source) => format!("invalid shift reference: {source}"),
                None => unreachable!("valid shift reference falsely detected as invalid"),
            },
            Kind::InvalidIntensities(positions) => {
                let length = positions.len();
                let indices = positions[..usize::min(5, length)]
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");

                match length {
                    0 => unreachable!("valid intensities falsely detected as invalid"),
                    1 => format!("intensities contains a non-finite value at index [{indices}]"),
                    2..=5 => {
                        format!("intensities contain non-finite values at indices [{indices}]")
                    }
                    _ => format!(
                        "intensities contain non-finite values at indices \
                         [{indices}, ...] ({length} invalid values)",
                    ),
                }
            }
        };

        write!(f, "{description}")
    }
}

impl Error {
    /// Creates a new [`NonFiniteFloat`] error.
    ///
    /// [`NonFiniteFloat`]: Kind::NonFiniteFloat
    pub(crate) fn non_finite_float() -> Self {
        Kind::NonFiniteFloat.into()
    }

    /// Creates a new [`EmptyData`] error.
    ///
    /// [`EmptyData`]: Kind::EmptyData
    pub(crate) fn empty_data() -> Self {
        Kind::EmptyData.into()
    }

    /// Creates a new [`InvalidRange`] error.
    ///
    /// [`InvalidRange`]: Kind::InvalidRange
    pub(crate) fn invalid_range() -> Self {
        Kind::InvalidRange.into()
    }

    /// Creates a new [`InvalidSpectrometerFrequency`] error.
    ///
    /// [`InvalidSpectrometerFrequency`]: Kind::InvalidLarmor
    pub(crate) fn invalid_larmor(source: Option<Self>) -> Self {
        Self {
            kind: Kind::InvalidLarmor,
            source: source.map(Arc::new),
        }
    }

    /// Creates a new [`InvalidSignalBoundaries`] error.
    ///
    /// [`InvalidSignalBoundaries`]: Kind::InvalidSignalBoundaries
    pub(crate) fn invalid_signal_boundaries(source: Self) -> Self {
        Self {
            kind: Kind::InvalidSignalBoundaries,
            source: Some(Arc::new(source)),
        }
    }

    /// Creates a new [`OutOfBounds`] error.
    ///
    /// [`OutOfBounds`]: Kind::OutOfBounds
    pub(crate) fn out_of_bounds() -> Self {
        Kind::OutOfBounds.into()
    }

    /// Creates a new [`InvalidShiftReference`] error.
    ///
    /// [`InvalidShiftReference`]: Kind::InvalidShiftReference
    pub(crate) fn invalid_shift_reference(source: Self) -> Self {
        Self {
            kind: Kind::InvalidShiftReference,
            source: Some(Arc::new(source)),
        }
    }

    /// Creates a new [`InvalidIntensities`] error.
    ///
    /// [`InvalidIntensities`]: Kind::InvalidIntensities
    pub(crate) fn invalid_intensities(positions: Vec<usize>) -> Self {
        Kind::InvalidIntensities(positions).into()
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// Returns the source of this error, if any.
    pub fn source(&self) -> Option<&Error> {
        self.source.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(Error: Send, Sync);
        assert_impl_all!(Kind: Send, Sync);
        assert_impl_all!(Result<()>: Send, Sync);
    }
}
