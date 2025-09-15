use std::sync::Arc;
use zeenmr_peakshape::PeakShape;

#[derive(Clone, Debug)]
pub struct Deconvolution<P> {
    peak_shapes: Arc<[P]>,
}

impl<P> Deconvolution<P>
where
    P: PeakShape,
{
    pub fn new<I>(peak_shapes: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self {
            peak_shapes: Arc::from_iter(peak_shapes),
        }
    }

    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }
}
