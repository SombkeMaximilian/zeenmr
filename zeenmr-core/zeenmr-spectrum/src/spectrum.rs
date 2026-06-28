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
pub struct Spectrum1D<S> {
    /// Nucleus observed in the NMR experiment.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    nucleus: Option<Nucleus>,
    /// Frequency axis of the spectrum.
    axis: Axis,
    /// Range within the intensities where signals are present.
    signal_range: Range<usize>,
    /// Intensity array or dual channel.
    intensities: S,
}

impl<S> Spectrum1D<S> {
    /// Returns the observed nucleus.
    pub fn nucleus(&self) -> Option<&Nucleus> {
        self.nucleus.as_ref()
    }

    /// Returns the frequency axis.
    pub fn axis(&self) -> &Axis {
        &self.axis
    }

    /// Returns the signal range.
    pub fn signal_range(&self) -> &Range<usize> {
        &self.signal_range
    }
}

impl<S> Spectrum1D<S>
where
    S: Array1D,
{
    /// Returns a slice containing the intensities.
    pub fn intensities(&self) -> &[S::Elem] {
        self.intensities.as_ref()
    }
}

impl<S1, S2> Spectrum1D<DualChannel1D<S1, S2>>
where
    S1: Array1D,
    S2: Array1D,
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

/// Post-initialization marker.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HasAxis;

/// Pre-initialization marker.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NeedsRange;

/// Post-initialization marker.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HasRange;

/// Builder for 1D spectra.
#[derive(Clone, PartialEq, Debug)]
pub struct Builder1D<S, K, A, R> {
    /// Nucleus observed in the NMR experiment.
    nucleus: Option<Nucleus>,
    /// Frequency axis of the spectrum.
    axis: Option<Axis>,
    /// Cached or overridden signal range.
    signal_range: Option<Range<usize>>,
    /// Intensity array or dual channel.
    intensities: S,
    /// State of the builder.
    state: (PhantomData<K>, PhantomData<A>, PhantomData<R>),
}

impl<S, K> Builder1D<S, K, HasAxis, HasRange>
where
    S: Array1D,
{
    /// Finalizes the spectrum.
    pub fn finalize(self) -> Spectrum1D<S> {
        Spectrum1D {
            nucleus: self.nucleus,
            axis: self
                .axis
                .expect("type state pattern should guarantee this works"),
            signal_range: self
                .signal_range
                .expect("type state pattern should guarantee this works"),
            intensities: self.intensities,
        }
    }
}

impl<S1, S2> Builder1D<DualChannel1D<S1, S2>, SingleChannel, HasAxis, HasRange>
where
    S1: Array1D,
    S2: Array1D<Elem = S1::Elem>,
{
    /// Finalizes the dual channel spectrum.
    pub fn finalize(self) -> Spectrum1D<DualChannel1D<S1, S2>> {
        Spectrum1D {
            nucleus: self.nucleus,
            axis: self
                .axis
                .expect("type state pattern should guarantee this works"),
            signal_range: self
                .signal_range
                .expect("type state pattern should guarantee this works"),
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
            axis: None,
            signal_range: None,
            intensities: array,
            state: (PhantomData, PhantomData, PhantomData),
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
            axis: None,
            signal_range: None,
            intensities: array,
            state: (PhantomData, PhantomData, PhantomData),
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
            axis: None,
            signal_range: None,
            intensities: DualChannel1D { real, imag },
            state: (PhantomData, PhantomData, PhantomData),
        })
    }
}

impl<S, K, R> Builder1D<S, K, NeedsAxis, R> {
    /// Sets the frequency axis.
    pub fn axis(self, axis: Axis) -> Builder1D<S, K, HasAxis, R> {
        Builder1D {
            nucleus: self.nucleus,
            axis: Some(axis),
            signal_range: self.signal_range,
            intensities: self.intensities,
            state: (self.state.0, PhantomData::<HasAxis>, self.state.2),
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
    pub fn signal_range<F>(self, finder: F) -> Result<Builder1D<S, K, A, HasRange>>
    where
        F: FindSignalRange<S::Elem, K>,
    {
        Ok(Builder1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range: Some(finder.find_signal_range(self.intensities.as_ref())?),
            intensities: self.intensities,
            state: (self.state.0, self.state.1, PhantomData::<HasRange>),
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
    ) -> Result<Builder1D<DualChannel1D<S1, S2>, SingleChannel, A, HasRange>>
    where
        F: FindSignalRange<S1::Elem, SingleChannel>,
    {
        let real_range = finder.find_signal_range(self.intensities.real.as_ref())?;
        let imag_range = finder.find_signal_range(self.intensities.imag.as_ref())?;
        let start = real_range.start.min(imag_range.start);
        let end = real_range.end.max(imag_range.end);
        let signal_range = Some(start..end);

        Ok(Builder1D {
            nucleus: self.nucleus,
            axis: self.axis,
            signal_range,
            intensities: self.intensities,
            state: (self.state.0, self.state.1, PhantomData::<HasRange>),
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
