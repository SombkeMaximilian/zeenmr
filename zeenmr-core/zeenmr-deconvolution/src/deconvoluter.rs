use crate::Deconvolution;
use crate::error::Result;
use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::marker::PhantomData;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum;

#[derive(Clone, Debug)]
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

    pub fn deconvolute(&self, spectrum: &Spectrum) -> Result<Deconvolution<P>> {
        let intensities = self.smoother.smooth(spectrum.intensities());
        let peaks = self.peak_finder.find_peaks(
            intensities,
            spectrum.signal_boundaries().unwrap(), // remove this later
            None,
        )?;
        let peak_shapes = self.fitter.fit_peak_shapes(spectrum, peaks);

        Ok(Deconvolution::new(peak_shapes))
    }
}
