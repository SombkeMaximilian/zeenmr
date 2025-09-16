use crate::Deconvoluter;
use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::sync::Arc;
use zeenmr_peakshape::PeakShape;

#[derive(Clone, PartialEq, Debug)]
pub struct Deconvolution<P, SMS, PFS, FTS> {
    smoothing_settings: SMS,
    peak_finding_settings: PFS,
    fitting_settings: FTS,
    peak_shapes: Arc<[P]>,
    mse: f64,
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
