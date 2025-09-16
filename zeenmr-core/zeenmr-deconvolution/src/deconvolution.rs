use crate::Deconvoluter;
use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::sync::Arc;
use zeenmr_peakshape::PeakShape;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of a deconvolution.
///
/// A `Deconvolution` contains the deconvoluted signals as peak shapes, the
/// settings used for deconvolution, and the mean squared error of between the
/// original [`Spectrum`] and the superposition of peak shapes.
///
/// [`Spectrum`]: zeenmr_spectrum::Spectrum
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Deconvolution` implements the
/// [`Serialize`] and [`Deserialize`] traits provided that the contained
/// types also implement them. This is guaranteed for any types provided by
/// this crate.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "P: Serialize,\
                     SMS: Serialize,\
                     PFS: Serialize,\
                     FTS: Serialize",
        deserialize = "P: Deserialize<'de>,\
                       SMS: Deserialize<'de>,\
                       PFS: Deserialize<'de>,\
                       FTS: Deserialize<'de>"
    ))
)]
pub struct Deconvolution<P, SMS, PFS, FTS> {
    /// Smoothing settings used.
    smoothing_settings: SMS,
    /// Peak finding settings used.
    peak_finding_settings: PFS,
    /// Fitting settings used.
    fitting_settings: FTS,
    /// Mean squared error of the deconvolution.
    mse: f64,
    /// Deconvoluted peak shapes.
    #[cfg_attr(feature = "serde", serde(with = "serialize_peak_shapes"))]
    peak_shapes: Arc<[P]>,
}

// Enabling the `rc` feature of serde would force it on everyone using this
// crate, so we just implement this manually for now. If `Arc` turns out to be
// unnecessary, we can switch to `Box<[P]>` later and remove this module.
#[cfg(feature = "serde")]
mod serialize_peak_shapes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub(crate) fn serialize<P, S>(peak_shapes: &Arc<[P]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        P: Serialize,
        S: Serializer,
    {
        peak_shapes.as_ref().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, P, D>(deserializer: D) -> Result<Arc<[P]>, D::Error>
    where
        P: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Box::<[P]>::deserialize(deserializer)?.into())
    }
}

impl<P, SMS, PFS, FTS> Deconvolution<P, SMS, PFS, FTS>
where
    P: PeakShape + Send + Sync,
{
    /// Creates a new `Deconvolution`.
    ///
    /// Normally, this type is only instantiated by the deconvolution functions
    /// of the [`Deconvoluter`] type.
    pub fn new<I, SM, PF, FT>(
        peak_shapes: I,
        deconvoluter: &Deconvoluter<P, SM, PF, FT>,
        mse: f64,
    ) -> Self
    where
        I: IntoIterator<Item = P>,
        SM: Smooth<Settings = SMS>,
        PF: FindPeaks<Settings = PFS>,
        FT: FitPeakShapes<P, Settings = FTS>,
    {
        Self {
            smoothing_settings: deconvoluter.smoothing_settings(),
            peak_finding_settings: deconvoluter.peak_finding_settings(),
            fitting_settings: deconvoluter.fitting_settings(),
            peak_shapes: Arc::from_iter(peak_shapes),
            mse,
        }
    }

    /// Returns the smoothing settings used in the deconvolution.
    pub fn smoothing_settings(&self) -> &SMS {
        &self.smoothing_settings
    }

    /// Returns the peak finding settings used in the deconvolution.
    pub fn peak_finding_settings(&self) -> &PFS {
        &self.peak_finding_settings
    }

    /// Returns the fitting settings used in the deconvolution.
    pub fn fitting_settings(&self) -> &FTS {
        &self.fitting_settings
    }

    /// Returns the deconvoluted peak shapes.
    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }

    /// Returns the mean squared error of the deconvolution.
    pub fn mse(&self) -> f64 {
        self.mse
    }
}
