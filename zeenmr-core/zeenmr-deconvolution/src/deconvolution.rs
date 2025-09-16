use crate::Deconvoluter;
use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::sync::Arc;
use zeenmr_peakshape::PeakShape;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Deconvolution<P, SMS, PFS, FTS> {
    smoothing_settings: SMS,
    peak_finding_settings: PFS,
    fitting_settings: FTS,
    mse: f64,
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

    pub fn smoothing_settings(&self) -> &SMS {
        &self.smoothing_settings
    }

    pub fn peak_finding_settings(&self) -> &PFS {
        &self.peak_finding_settings
    }

    pub fn fitting_settings(&self) -> &FTS {
        &self.fitting_settings
    }

    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }

    pub fn mse(&self) -> f64 {
        self.mse
    }
}
