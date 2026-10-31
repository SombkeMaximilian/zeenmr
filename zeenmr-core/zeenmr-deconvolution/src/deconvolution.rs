use crate::fitting::FittedComponents;
use num_traits::Float;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::dimension::DimIndex;
use zeenmr_util::StableGrid;

/// Deconvolution output.
#[derive(Clone, Debug)]
pub struct Deconvolution<'s, T, P> {
    /// Deconvoluted spectrum.
    spectrum: SpectrumView1D<'s, T, T>,
    /// Components of the spectrum.
    components: FittedComponents<T, P>,
}

impl<'s, T, P> Deconvolution<'s, T, P>
where
    T: Float,
    P: PeakShape<T>,
{
    /// Creates a new deconvolution.
    pub fn new(spectrum: SpectrumView1D<'s, T, T>, components: FittedComponents<T, P>) -> Self {
        Self {
            spectrum,
            components,
        }
    }

    /// Returns the spectrum view and its fitted components.
    pub fn into_parts(self) -> (SpectrumView1D<'s, T, T>, FittedComponents<T, P>) {
        (self.spectrum, self.components)
    }

    /// Returns the peak shapes in frequency space.
    pub fn into_frequency_space(self) -> Vec<P> {
        self.components.into_frequency_space(
            self.spectrum
                .axis(DimIndex(0))
                .expect("1D spectrum always has a first dimension"),
        )
    }

    /// Returns the peak shapes in shift space.
    pub fn into_shift_space(self) -> Vec<P> {
        self.components.into_shift_space(
            self.spectrum
                .axis(DimIndex(0))
                .expect("1D spectrum always has a first dimension"),
        )
    }

    /// Returns a view of the spectrum.
    pub fn spectrum(&self) -> SpectrumView1D<'_, T, T> {
        self.spectrum.view()
    }

    /// Returns the fitted components
    pub fn components(&self) -> &FittedComponents<T, P> {
        &self.components
    }

    /// Returns the deconvoluted peak shapes in fitting grid space.
    pub fn peak_shapes(&self) -> &[P] {
        &self.components.peak_shapes()
    }

    /// Returns the fitting grid.
    pub fn grid(&self) -> StableGrid<T> {
        self.components.grid()
    }

    /// Returns the peak shapes in frequency space.
    ///
    /// Note that this clones the data. Prefer
    /// [`Deconvolution::into_frequency_space`] to consume `self` instead.
    pub fn frequency_space(&self) -> Vec<P>
    where
        P: Clone,
    {
        self.components.frequency_space(
            self.spectrum
                .axis(DimIndex(0))
                .expect("1D spectrum always has a first dimension"),
        )
    }

    /// Returns the peak shapes in shift space.
    ///
    /// Note that this clones the data. Prefer
    /// [`Deconvolution::into_shift_space`] to consume `self` instead.
    pub fn shift_space(&self) -> Vec<P>
    where
        P: Clone,
    {
        self.components.shift_space(
            self.spectrum
                .axis(DimIndex(0))
                .expect("1D spectrum always has a first dimension"),
        )
    }
}
