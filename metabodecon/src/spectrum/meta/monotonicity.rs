/// Represents the ordering of 1D NMR spectrum data.
///
/// Typically, 1D NMR data is ordered in `Decreasing` order of chemical shifts,
/// but this is not always the case. Additionally, it is often simpler to work
/// with the data if it is ordered in `Increasing` order, and only reorder it
/// for display purposes.
///
/// `Monotonicity` is not intended to be instantiated directly. Instead, it is
/// a marker used to indicate the ordering of data in a [`Spectrum`].
///
/// [`Spectrum`]: crate::spectrum::Spectrum
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Monotonicity {
    /// Data is ordered in increasing order of chemical shifts.
    #[default]
    Increasing,
    /// Data is ordered in decreasing order of chemical shifts.
    Decreasing,
}

impl Monotonicity {
    /// Helper function to determine the `Monotonicity` from 2 floating point
    /// numbers.
    ///
    /// Checks for the ordering of two floating point numbers and returns the
    /// corresponding `Some(Monotonicity)` variant. If the two numbers differ by
    /// less than a small multiple of the floating point precision, or are not
    /// finite numbers, or cannot be compared, `None` is returned.
    pub(crate) fn from_f64s(first: f64, second: f64) -> Option<Self> {
        if f64::abs(first - second) < crate::CHECK_PRECISION || !(first - second).is_finite() {
            return None;
        }
        match first.partial_cmp(&second) {
            Some(std::cmp::Ordering::Less) => Some(Self::Increasing),
            Some(std::cmp::Ordering::Greater) => Some(Self::Decreasing),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(Monotonicity);
        assert_sync!(Monotonicity);
    }

    #[test]
    fn increasing() {
        let first = 3.0;
        let second = 4.0;
        assert_eq!(
            Monotonicity::from_f64s(first, second),
            Some(Monotonicity::Increasing)
        );
    }

    #[test]
    fn decreasing() {
        let first = 4.0;
        let second = 3.0;
        assert_eq!(
            Monotonicity::from_f64s(first, second),
            Some(Monotonicity::Decreasing)
        );
    }

    #[test]
    fn indeterminate() {
        let firsts = [3.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let seconds = [3.0 + f64::EPSILON, 0.0, 0.0, 0.0];
        firsts
            .into_iter()
            .zip(seconds)
            .for_each(|(first, second)| assert_eq!(Monotonicity::from_f64s(first, second), None));
    }
}
