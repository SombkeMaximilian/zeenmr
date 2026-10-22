#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

pub mod iter;

pub mod axis;

pub mod dimension;

pub mod intensity_array;

mod nucleus;
pub use nucleus::Nucleus;

mod spectrum;
pub use spectrum::{
    Spectrum, Spectrum1D, Spectrum2D, Spectrum3D, SpectrumArc, SpectrumArc1D, SpectrumArc2D,
    SpectrumArc3D, SpectrumArcDyn, SpectrumCow, SpectrumCow1D, SpectrumCow2D, SpectrumCow3D,
    SpectrumCowDyn, SpectrumCowSlice, SpectrumCowSlice1D, SpectrumCowSlice2D, SpectrumCowSlice3D,
    SpectrumCowSliceDyn, SpectrumDyn, SpectrumOwned, SpectrumOwned1D, SpectrumOwned2D,
    SpectrumOwned3D, SpectrumOwnedDyn, SpectrumRc, SpectrumRc1D, SpectrumRc2D, SpectrumRc3D,
    SpectrumRcDyn, SpectrumSliceView, SpectrumSliceView1D, SpectrumSliceView2D,
    SpectrumSliceView3D, SpectrumSliceViewDyn, SpectrumView, SpectrumView1D, SpectrumView2D,
    SpectrumView3D, SpectrumViewDyn,
};
