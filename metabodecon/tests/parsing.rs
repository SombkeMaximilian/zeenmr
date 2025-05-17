use float_cmp::assert_approx_eq;
use metabodecon::spectrum::*;

pub mod utilities;
use utilities::{PRECISION, workspace_dir};

#[test]
fn parse_blood() {
    let data_dir = workspace_dir().join("data");
    let bruker_path = data_dir.join("bruker").join("blood");
    let jcampdx_path = data_dir.join("jcamp-dx").join("blood");
    let mut bruker_spectra = Bruker::read_spectra(bruker_path, 10, 10, (-2.2, 11.8)).unwrap();
    let mut jcampdx_spectra = JcampDx::read_spectra(jcampdx_path, (-2.2, 11.8)).unwrap();

    // The order the spectra are read in depends on the OS, we have to sort them
    bruker_spectra.sort_by(|a, b| {
        a.intensities()[0]
            .partial_cmp(&b.intensities()[0])
            .unwrap()
    });
    jcampdx_spectra.sort_by(|a, b| {
        a.intensities()[0]
            .partial_cmp(&b.intensities()[0])
            .unwrap()
    });

    bruker_spectra
        .iter()
        .zip(jcampdx_spectra.iter())
        .for_each(|(bruker, jcampdx)| {
            bruker
                .chemical_shifts()
                .iter()
                .zip(jcampdx.chemical_shifts().iter())
                .for_each(|(bx, jx)| {
                    assert_approx_eq!(f64, *bx, *jx, epsilon = PRECISION);
                });
            bruker
                .intensities()
                .iter()
                .zip(jcampdx.intensities().iter())
                .for_each(|(bi, ji)| {
                    assert_approx_eq!(f64, *bi, *ji, epsilon = PRECISION);
                });
            assert_eq!(bruker.nucleus(), jcampdx.nucleus());
            assert_approx_eq!(
                f64,
                bruker.frequency(),
                jcampdx.frequency(),
                epsilon = PRECISION
            );
            assert_approx_eq!(
                f64,
                bruker.reference_compound().chemical_shift(),
                jcampdx.reference_compound().chemical_shift(),
                epsilon = PRECISION
            );
            assert_eq!(
                bruker.reference_compound().index(),
                jcampdx.reference_compound().index()
            );
        });
}
