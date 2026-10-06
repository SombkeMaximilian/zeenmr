//! Types for estimating noise levels.

use crate::util::signal_boundaries::SignalBoundaries;
use num_traits::Float;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::axis::range::RelativeRange;
use zeenmr_spectrum::dimension::DimIndex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Intensity noise level.
///
/// Any intensity value of a lower magnitude cannot be discerned from noise.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize), serde(transparent))]
#[repr(transparent)]
pub struct Noise<T>(pub T);

/// Trait for estimating the noise level of a spectrum.
pub trait NoiseLevel<T> {
    /// Returns an estimate for the noise level of the spectrum.
    fn noise_level(&self, spectrum: SpectrumView1D<T, T>) -> Noise<T>;
}

/// Estimates Gaussian distribution parameters from the non-signal part of the
/// spectrum.
///
/// The computed noise-level is the mean plus a multiple of the standard
/// deviation.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct GaussianNoise<T, S> {
    /// Multiple of the standard deviation.
    threshold: T,
    /// Type determining the signal boundaries of the input spectrum.
    signal_finder: S,
}

impl<T> Default for GaussianNoise<T, RelativeRange<T>>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            threshold: T::from(5).expect("conversion from u8 to T must never fail"),
            signal_finder: RelativeRange::new(
                T::from(0.2).expect("conversion from {float} to T must never fail"),
                T::from(0.8).expect("conversion from {float} to T must never fail"),
            )
            .expect("bounds are in [0, 1]"),
        }
    }
}

impl<T, S> NoiseLevel<T> for GaussianNoise<T, S>
where
    T: Float,
    S: SignalBoundaries<T>,
{
    fn noise_level(&self, spectrum: SpectrumView1D<T, T>) -> Noise<T> {
        let signal = self
            .signal_finder
            .signal_boundaries(spectrum.view());
        let intensities = spectrum.intensities();
        let left = intensities
            .cropped(DimIndex(0), ..signal.start)
            .expect("1D spectrum always has a first dimension");
        let right = intensities
            .cropped(DimIndex(0), signal.end..)
            .expect("1D spectrum always has a first dimension");
        let len =
            T::from(left.len() + right.len()).expect("conversion from usize to T must never fail");
        let mean = (left.elem().fold(T::zero(), |acc, &x| acc + x)
            + right.elem().fold(T::zero(), |acc, &x| acc + x))
            / len;
        let variance = (left
            .elem()
            .fold(T::zero(), |acc, &x| acc + (x - mean).powi(2))
            + right
                .elem()
                .fold(T::zero(), |acc, &x| acc + (x - mean).powi(2)))
            / len;

        Noise(mean + self.threshold * variance.sqrt())
    }
}

impl<T, S> GaussianNoise<T, S> {
    /// Creates a new Gaussian noise estimator.
    pub fn new(threshold: T, signal_finder: S) -> Self {
        Self {
            threshold,
            signal_finder,
        }
    }
}
