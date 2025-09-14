use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::marker::PhantomData;
use zeenmr_peakshape::PeakShape;

#[derive(Clone)]
pub struct Deconvoluter<P, SM, PF, FT> {
    smoother: SM,
    peak_finder: PF,
    fitter: FT,
    peak_shape: PhantomData<P>,
}

impl<P, SM, PF, FT> Deconvoluter<P, SM, PF, FT>
where
    P: PeakShape + Send + Sync,
    SM: Smooth + Send + Sync,
    PF: FindPeaks + Send + Sync,
    FT: FitPeakShapes<P> + Send + Sync,
{
    pub fn new(smoother: SM, peak_finder: PF, fitter: FT) -> Self {
        Self {
            smoother,
            peak_finder,
            fitter,
            peak_shape: PhantomData::<P>,
        }
    }
}
