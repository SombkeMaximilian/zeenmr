use metabodecon::deconvolution::*;
use metabodecon::spectrum::*;

pub mod utilities;
use utilities::{store_deconvolution, workspace_dir};

#[test]
fn sim() {
    let path = workspace_dir()
        .join("data")
        .join("bruker")
        .join("sim")
        .join("sim_01");
    let spectrum = Bruker::read_spectrum(path, 10, 10, (3.35, 3.55)).unwrap();
    let deconvoluter = Deconvoluter::default();
    let deconvolution = deconvoluter
        .deconvolute_spectrum(&spectrum)
        .unwrap();
    let filename = format!("{}_deconvolution.json", "sim_01");
    store_deconvolution(deconvolution, filename.as_str());
}

#[test]
fn blood() {
    let path = workspace_dir()
        .join("data")
        .join("bruker")
        .join("blood")
        .join("blood_01");
    let spectrum = Bruker::read_spectrum(path, 10, 10, (-2.2, 11.8)).unwrap();
    let deconvoluter = Deconvoluter::default();
    let deconvolution = deconvoluter
        .deconvolute_spectrum(&spectrum)
        .unwrap();
    let filename = format!("{}_deconvolution.json", "blood_01");
    store_deconvolution(deconvolution, filename.as_str());
}

#[cfg(feature = "parallel")]
#[test]
fn par_sim() {
    let path = workspace_dir()
        .join("data")
        .join("bruker")
        .join("sim")
        .join("sim_01");
    let spectrum = Bruker::read_spectrum(path, 10, 10, (3.35, 3.55)).unwrap();
    let deconvoluter = Deconvoluter::default();
    let deconvolution = deconvoluter
        .par_deconvolute_spectrum(&spectrum)
        .unwrap();
    let filename = format!("{}_par_deconvolution.json", "sim_01");
    store_deconvolution(deconvolution, filename.as_str());
}

#[cfg(feature = "parallel")]
#[test]
fn par_blood() {
    let path = workspace_dir()
        .join("data")
        .join("bruker")
        .join("blood")
        .join("blood_01");
    let spectrum = Bruker::read_spectrum(path, 10, 10, (-2.2, 11.8)).unwrap();
    let deconvoluter = Deconvoluter::default();
    let deconvolution = deconvoluter
        .par_deconvolute_spectrum(&spectrum)
        .unwrap();
    let filename = format!("{}_par_deconvolution.json", "blood_01");
    store_deconvolution(deconvolution, filename.as_str());
}
