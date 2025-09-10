use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use num_traits::Float;
use std::iter::Sum;
use std::marker::PhantomData;
use zeenmr_peakshape::PeakShape;

#[derive(Clone)]
pub struct Deconvoluter<F, P, SM, PF, FT> {
    smoother: SM,
    peak_finder: PF,
    fitter: FT,
    float_type: PhantomData<F>,
    peak_shape: PhantomData<P>,
}

impl<F, P, SM, PF, FT> Deconvoluter<F, P, SM, PF, FT>
where
    F: Float + Sum,
    P: PeakShape,
    SM: Smooth<F> + Send + Sync,
    PF: FindPeaks<F> + Send + Sync,
    FT: FitPeakShapes<F, P> + Send + Sync,
{
    pub fn new(smoother: SM, peak_finder: PF, fitter: FT) -> Self {
        Self {
            smoother,
            peak_finder,
            fitter,
            float_type: PhantomData::<F>,
            peak_shape: PhantomData::<P>,
        }
    }
}
