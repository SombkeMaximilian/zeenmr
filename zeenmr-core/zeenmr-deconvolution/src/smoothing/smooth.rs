use crate::error::EitherError;
use std::borrow::Cow;

/// Trait for smoothing algorithms that process a sequence of values.
///
/// # Note for Implementors
///
/// It is not necessary to handle floating point `NaN` or either infinity. The
/// passed `data` can be assumed to contain no such value.
///
/// If no smoothing can be performed because the data does not contain enough
/// elements, the `Ok` variant should be returned without performing any
/// smoothing.
pub trait Smooth<T>
where
    T: Clone,
{
    /// Error type when an error occurs during smoothing.
    type Error;

    /// Smooth the provided data in place.
    fn smooth_in_place(&self, data: &mut [T]) -> Result<(), Self::Error>;

    /// Smooth the provided data and return an owned instance of the smoothed
    /// values.
    fn smooth<'a>(&self, data: &'a [T]) -> Result<Cow<'a, [T]>, Self::Error> {
        let mut data = data.to_owned();
        self.smooth_in_place(&mut data)?;

        Ok(data.into())
    }

    /// Returns a smoother that first applies this smoother and then the other.
    fn chain<S>(self, other: S) -> Chain<Self, S>
    where
        Self: Sized,
        S: Smooth<T>,
    {
        Chain {
            first: self,
            second: other,
        }
    }
}

/// No-op smoother that does not modify the input values.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Identity;

impl<T> Smooth<T> for Identity
where
    T: Clone,
{
    type Error = std::convert::Infallible;

    fn smooth_in_place(&self, _: &mut [T]) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub fn new() -> Self {
        Self
    }
}

/// Smoother that applies its two components to the data in sequence.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Chain<S1, S2> {
    /// First smoother to apply.
    first: S1,
    /// Second smoother to apply.
    second: S2,
}

impl<T, S1, S2> Smooth<T> for Chain<S1, S2>
where
    T: Clone,
    S1: Smooth<T>,
    S2: Smooth<T>,
{
    type Error = EitherError<S1::Error, S2::Error>;

    fn smooth_in_place(&self, data: &mut [T]) -> Result<(), Self::Error> {
        self.first
            .smooth_in_place(data)
            .map_err(EitherError::First)?;
        self.second
            .smooth_in_place(data)
            .map_err(EitherError::Second)?;
        Ok(())
    }
}
