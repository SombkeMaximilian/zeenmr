mod fitter;
pub(crate) use fitter::Fitter;
pub use fitter::FittingSettings;

mod fitter_analytical;
pub(crate) use fitter_analytical::FitterAnalytical;

mod peak_stencil;
pub(crate) use peak_stencil::PeakStencil;

mod reduced_spectrum;
pub(crate) use reduced_spectrum::ReducedSpectrum;
