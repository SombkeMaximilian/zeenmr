use crate::Nucleus;
use crate::error::Result;
use crate::frequency_axis::Axis;
use crate::intensity_array::Array1D;
use crate::intensity_array::diagnostic_1d::{
    FindSignalRange, Magnitude, SingleChannel, ValidateIntensities,
};
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Wrapper around two 1D arrays for real and imaginary channels.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct DualChannel1D<S1, S2> {
    /// Real, absorptive channel.
    pub(crate) real: S1,
    /// Imaginary, dispersive channel.
    pub(crate) imag: S2,
}

/// 1D NMR spectrum.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct Spectrum1D<T, S> {
    /// Nucleus observed in the NMR experiment.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    nucleus: Option<Nucleus>,
    /// Frequency axis of the spectrum.
    axis: Axis<T>,
    /// Range within the intensities where signals are present.
    signal_range: Range<usize>,
    /// Intensity array or dual channel.
    intensities: S,
}

impl<T, S> Spectrum1D<T, S> {
    /// Returns the observed nucleus.
    pub fn nucleus(&self) -> Option<&Nucleus> {
        self.nucleus.as_ref()
    }

    /// Returns the frequency axis.
    pub fn axis(&self) -> &Axis<T> {
        &self.axis
    }

    /// Returns the signal range.
    pub fn signal_range(&self) -> &Range<usize> {
        &self.signal_range
    }
}

impl<S> Spectrum1D<S::Elem, S>
where
    S: Array1D,
{
    /// Returns a slice containing the intensities.
    pub fn intensities(&self) -> &[S::Elem] {
        self.intensities.as_ref()
    }
}

impl<S1, S2> Spectrum1D<S1::Elem, DualChannel1D<S1, S2>>
where
    S1: Array1D,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Returns a slice containing the real channel intensities.
    pub fn real(&self) -> &[S1::Elem] {
        self.intensities.real.as_ref()
    }

    /// Returns a slice containing the imaginary channel intensities.
    pub fn imag(&self) -> &[S2::Elem] {
        self.intensities.imag.as_ref()
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
pub struct Builder1D<S, K, A, R> {
    /// Nucleus observed in the NMR experiment.
    nucleus: Option<Nucleus>,
    /// Frequency axis of the spectrum.
    axis: A,
    /// Cached or overridden signal range.
    signal_range: R,
    /// Intensity array or dual channel.
    intensities: S,
    /// Magnitude or single channel intensities.
    intensity_kind: PhantomData<K>,
}

impl<S, K> Builder1D<S, K, Axis<S::Elem>, Range<usize>>
where
    S: Array1D,
{
    /// Finalizes the spectrum.
    pub fn finalize(self) -> Spectrum1D<S::Elem, S> {
        Spectrum1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
        }
    }
}

impl<S1, S2> Builder1D<DualChannel1D<S1, S2>, SingleChannel, Axis<S1::Elem>, Range<usize>>
where
    S1: Array1D,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Finalizes the dual channel spectrum.
    pub fn finalize(self) -> Spectrum1D<S1::Elem, DualChannel1D<S1, S2>> {
        Spectrum1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
        }
    }
}

impl<S> Builder1D<S, Magnitude, NeedsAxis, NeedsRange>
where
    S: Array1D,
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
            nucleus: None,
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: array,
            intensity_kind: PhantomData,
        })
    }
}

impl<S> Builder1D<S, SingleChannel, NeedsAxis, NeedsRange>
where
    S: Array1D,
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
            nucleus: None,
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: array,
            intensity_kind: PhantomData,
        })
    }
}

impl<S1, S2> Builder1D<DualChannel1D<S1, S2>, SingleChannel, NeedsAxis, NeedsRange>
where
    S1: Array1D,
    S1::Elem: Float,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Build a dual channel spectrum with real and imaginary intensities.
    ///
    /// # Errors
    ///
    /// Returns an error if validating either intensities fails.
    pub fn dual_channel(real: S1, imag: S2) -> Result<Self> {
        SingleChannel::validate(&real)?;
        SingleChannel::validate(&imag)?;

        Ok(Self {
            nucleus: None,
            axis: NeedsAxis,
            signal_range: NeedsRange,
            intensities: DualChannel1D { real, imag },
            intensity_kind: PhantomData,
        })
    }
}

impl<S, K, R> Builder1D<S, K, NeedsAxis, R>
where
    S: Array1D,
{
    /// Sets the frequency axis.
    pub fn axis(self, axis: Axis<S::Elem>) -> Builder1D<S, K, Axis<S::Elem>, R> {
        Builder1D {
            nucleus: self.nucleus,
            axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
            intensity_kind: self.intensity_kind,
        }
    }
}

impl<S1, S2, K, R> Builder1D<DualChannel1D<S1, S2>, K, NeedsAxis, R>
where
    S1: Array1D,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Sets the frequency axis.
    pub fn axis(
        self,
        axis: Axis<S1::Elem>,
    ) -> Builder1D<DualChannel1D<S1, S2>, K, Axis<S1::Elem>, R> {
        Builder1D {
            nucleus: self.nucleus,
            axis,
            signal_range: self.signal_range,
            intensities: self.intensities,
            intensity_kind: self.intensity_kind,
        }
    }
}

impl<S, K, A> Builder1D<S, K, A, NeedsRange>
where
    S: Array1D,
    S::Elem: Float,
{
    /// Sets the signal range with a finder.
    ///
    /// To directly set the signal range, `Range<usize>` is also a finder, which
    /// simply returns itself.
    ///
    /// # Errors
    ///
    /// Returns an error if finding the signal range fails.
    pub fn signal_range<F>(self, finder: F) -> Result<Builder1D<S, K, A, Range<usize>>>
    where
        F: FindSignalRange<S::Elem, K>,
    {
        Ok(Builder1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range: finder.find_signal_range(self.intensities.as_ref())?,
            intensities: self.intensities,
            intensity_kind: self.intensity_kind,
        })
    }
}

impl<S1, S2, A> Builder1D<DualChannel1D<S1, S2>, SingleChannel, A, NeedsRange>
where
    S1: Array1D,
    S1::Elem: Float,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Sets the signal range with a finder.
    ///
    /// Uses the union of the signal ranges found in the real and imaginary
    /// arrays.
    ///
    /// To directly set the signal range, `Range<usize>` is also a finder, which
    /// simply returns itself.
    ///
    /// # Errors
    ///
    /// Returns an error if finding the signal range fails.
    pub fn signal_range<F>(
        self,
        finder: F,
    ) -> Result<Builder1D<DualChannel1D<S1, S2>, SingleChannel, A, Range<usize>>>
    where
        F: FindSignalRange<S1::Elem, SingleChannel>,
    {
        let real_range = finder.find_signal_range(self.intensities.real.as_ref())?;
        let imag_range = finder.find_signal_range(self.intensities.imag.as_ref())?;
        let start = real_range.start.min(imag_range.start);
        let end = real_range.end.max(imag_range.end);
        let signal_range = start..end;

        Ok(Builder1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range,
            intensities: self.intensities,
            intensity_kind: self.intensity_kind,
        })
    }
}

impl<S, K, A, R> Builder1D<S, K, A, R> {
    /// Sets the observed nucleus.
    pub fn nucleus(mut self, nucleus: Nucleus) -> Self {
        self.nucleus = Some(nucleus);

        self
    }
}
