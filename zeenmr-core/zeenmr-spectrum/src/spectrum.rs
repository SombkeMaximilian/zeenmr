use crate::axis::FrequencyAxis;
use crate::error::Result;
use crate::intensity_array::Storage;
use crate::intensity_array::diagnostic_1d::{
    DualChannel, FindSignalRange, Magnitude, SingleChannel, ValidateIntensities,
};
use num_complex::Complex;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 1D NMR spectrum.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        rename_all = "camelCase",
        bound(deserialize = "T: Float + Deserialize<'de>, S: Deserialize<'de>")
    )
)]
pub struct Spectrum1D<T, S> {
    /// Frequency axis of the spectrum.
    axis: FrequencyAxis<T>,
    /// Range within the intensities where signals are present.
    signal_range: Range<usize>,
    /// Intensity array.
    intensities: S,
}

impl<T, S> Spectrum1D<T, S> {
    /// Returns the frequency axis.
    pub fn axis(&self) -> &FrequencyAxis<T> {
        &self.axis
    }

    /// Returns the signal range.
    pub fn signal_range(&self) -> &Range<usize> {
        &self.signal_range
    }
}

impl<T, S> Spectrum1D<T, S>
where
    S: Storage,
{
    /// Returns a slice containing the intensities.
    pub fn intensities(&self) -> &[S::Elem] {
        self.intensities.as_ref()
    }
}

impl<T, S> Spectrum1D<T, S>
where
    T: Copy,
    S: Storage,
{
    /// Returns a borrowed view of this spectrum.
    pub fn view(&self) -> SpectrumView1D<'_, T, S::Elem> {
        SpectrumView1D {
            axis: self.axis,
            signal_range: self.signal_range.clone(),
            intensities: self.intensities.as_ref(),
        }
    }
}

/// A borrowed view of a 1D spectrum.
///
/// Note that the element type may differ from the scalar type, but will always
/// be coupled to it in some way (e.g., when `E = Complex<T>`).
pub type SpectrumView1D<'s, T, E> = Spectrum1D<T, &'s [E]>;

impl<T, E> SpectrumView1D<'_, T, E>
where
    T: Copy,
    E: Clone,
{
    /// Returns an owned spectrum by copying the intensities of this view.
    pub fn to_owned(&self) -> Spectrum1D<T, Vec<E>> {
        Spectrum1D {
            axis: self.axis,
            signal_range: self.signal_range.clone(),
            intensities: self.intensities.to_vec(),
        }
    }
}

/// Pre-initialization marker.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NeedsAxis;

/// Pre-initialization marker.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NeedsRange;

/// Builder for 1D spectra.
#[derive(Clone, PartialEq, Debug)]
pub struct Builder1D<T, S, K, A, R> {
    /// Frequency axis of the spectrum.
    axis: A,
    /// Cached or overridden signal range.
    signal_range: R,
    /// Intensity array or dual channel.
    intensities: S,
    /// Scalar type.
    scalar: PhantomData<T>,
    /// Magnitude, single or dual channel intensities.
    intensity_kind: PhantomData<K>,
}

impl<T, S, K> Builder1D<T, S, K, FrequencyAxis<T>, Range<usize>>
where
    S: Storage,
{
    /// Finalizes the spectrum.
    pub fn finalize(self) -> Spectrum1D<T, S> {
        Spectrum1D {
            axis: self.axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
        }
    }
}

impl<S> Builder1D<S::Elem, S, Magnitude, NeedsAxis, NeedsRange>
where
    S: Storage,
    S::Elem: Float,
{
    /// Build a magnitude spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if validating the intensities fails.
    pub fn magnitude(array: S) -> Result<Self> {
        Magnitude::validate(&array)?;

        Ok(Self {
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: array,
            scalar: PhantomData,
            intensity_kind: PhantomData,
        })
    }
}

impl<S> Builder1D<S::Elem, S, SingleChannel, NeedsAxis, NeedsRange>
where
    S: Storage,
    S::Elem: Float,
{
    /// Build a single channel, real spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if validating the intensities fails.
    pub fn real(array: S) -> Result<Self> {
        SingleChannel::validate(&array)?;

        Ok(Self {
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: array,
            scalar: PhantomData,
            intensity_kind: PhantomData,
        })
    }

    /// Build a single channel, imaginary spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if validating the intensities fails.
    pub fn imag(array: S) -> Result<Self> {
        Self::real(array)
    }
}

impl<T, S> Builder1D<T, S, DualChannel, NeedsAxis, NeedsRange>
where
    T: Float,
    S: Storage<Elem = Complex<T>>,
{
    /// Build a dual channel, complex spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error if validating the intensities fails.
    pub fn complex(array: S) -> Result<Self> {
        DualChannel::validate(&array)?;

        Ok(Self {
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: array,
            scalar: PhantomData,
            intensity_kind: PhantomData,
        })
    }
}

impl<T, S, K, R> Builder1D<T, S, K, NeedsAxis, R>
where
    S: Storage,
{
    /// Sets the frequency axis.
    pub fn axis(self, axis: FrequencyAxis<T>) -> Builder1D<T, S, K, FrequencyAxis<T>, R> {
        Builder1D {
            axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
            scalar: PhantomData,
            intensity_kind: self.intensity_kind,
        }
    }
}

impl<T, S, K, A> Builder1D<T, S, K, A, NeedsRange>
where
    S: Storage,
{
    /// Sets the signal range with a finder.
    ///
    /// To directly set the signal range, `Range<usize>` is also a finder, which
    /// simply returns itself.
    ///
    /// # Errors
    ///
    /// Returns an error if finding the signal range fails.
    pub fn signal_range<F>(self, finder: F) -> Result<Builder1D<T, S, K, A, Range<usize>>>
    where
        F: FindSignalRange<S::Elem, K>,
    {
        Ok(Builder1D {
            axis: self.axis,
            signal_range: finder.find_signal_range(self.intensities.as_ref())?,
            intensities: self.intensities,
            scalar: PhantomData,
            intensity_kind: self.intensity_kind,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axis::ShiftReference;
    use crate::range::FrequencyRange;

    fn valid_axis<T>() -> FrequencyAxis<T>
    where
        T: Float,
    {
        let start = T::zero();
        let end = T::from(12000_u32).unwrap();
        let larmor = T::from(600.25_f64).unwrap();
        let ref_freq = T::from(3000_u32).unwrap();

        FrequencyAxis::new(
            FrequencyRange::new(start, end).unwrap(),
            larmor,
            ShiftReference::from_freq(ref_freq).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn real_spectrum() {
        fn _real_spectrum<T>() -> Result<()>
        where
            T: Float + std::fmt::Debug,
        {
            let data = (0..2_u32.pow(8))
                .map(|x| T::from(x).unwrap())
                .collect::<Vec<_>>();
            let range = 2_usize.pow(4)..(data.len() - 2_usize.pow(4));
            let axis = valid_axis();
            let spectrum = Builder1D::real(data)?
                .axis(axis)
                .signal_range(range)?
                .finalize();

            let _ = spectrum.axis();
            let _ = spectrum.signal_range();
            let _ = spectrum.intensities();

            Ok(())
        }

        _real_spectrum::<f32>().unwrap();
        _real_spectrum::<f64>().unwrap();
    }

    #[test]
    fn magnitude_spectrum() {
        fn _magnitude_spectrum<T>() -> Result<()>
        where
            T: Float + std::fmt::Debug,
        {
            let data = (0..2_u32.pow(8))
                .map(|x| T::from(x).unwrap())
                .collect::<Vec<T>>();
            let range = 2_usize.pow(4)..(data.len() - 2_usize.pow(4));
            let axis = valid_axis();
            let spectrum = Builder1D::magnitude(data)?
                .axis(axis)
                .signal_range(range)?
                .finalize();

            let _ = spectrum.axis();
            let _ = spectrum.signal_range();
            let _ = spectrum.intensities();

            Ok(())
        }

        _magnitude_spectrum::<f32>().unwrap();
        _magnitude_spectrum::<f64>().unwrap();
    }

    #[test]
    fn complex_spectrum() {
        fn _complex_spectrum<T>() -> Result<()>
        where
            T: Float + std::fmt::Debug,
        {
            let data = (0..2_u32.pow(8))
                .map(|x| T::from(x).unwrap())
                .map(|x| Complex::new(x, x))
                .collect::<Vec<Complex<T>>>();
            let range = 2_usize.pow(4)..(data.len() - 2_usize.pow(4));
            let axis = valid_axis();
            let spectrum = Builder1D::complex(data)?
                .axis(axis)
                .signal_range(range)?
                .finalize();

            let _ = spectrum.axis();
            let _ = spectrum.signal_range();
            let _ = spectrum.intensities();

            Ok(())
        }

        _complex_spectrum::<f32>().unwrap();
        _complex_spectrum::<f64>().unwrap();
    }
}
