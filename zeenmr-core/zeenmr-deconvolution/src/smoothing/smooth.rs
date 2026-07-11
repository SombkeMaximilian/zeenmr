use std::borrow::Cow;

/// Trait for smoothing algorithms that process a sequence of values.
pub trait Smooth<T>
where
    T: Clone,
{
    /// Smooth the provided data in place.
    fn smooth_in_place(&self, data: &mut [T]);

    /// Smooth the provided data and return an owned instance of the smoothed
    /// values.
    fn smooth<'a>(&self, data: &'a [T]) -> Cow<'a, [T]> {
        let mut data = data.to_owned();
        self.smooth_in_place(&mut data);

        data.into()
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
    fn smooth_in_place(&self, _: &mut [T]) {}
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
    fn smooth_in_place(&self, data: &mut [T]) {
        self.first.smooth_in_place(data);
        self.second.smooth_in_place(data);
    }
}
