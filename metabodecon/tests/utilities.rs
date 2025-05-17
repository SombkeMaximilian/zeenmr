use metabodecon::deconvolution::Deconvolution;
use std::path::PathBuf;

// A lot of the conversions are already imprecise before the spectra are parsed,
// so the required precision here is fairly lenient.
pub const PRECISION: f64 = 1.0e-3;

pub fn workspace_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn store_deconvolution(deconvolution: Deconvolution, filename: &str) {
    let tmp_path = env!("CARGO_TARGET_TMPDIR");
    let deconvolutions_dir = format!("{}/test-deconvolutions", tmp_path);
    let filename = format!("{}/test-deconvolutions/{}", tmp_path, filename);
    let serialized = serde_json::to_string_pretty(deconvolution.as_ref()).unwrap();
    std::fs::create_dir_all(&deconvolutions_dir).unwrap();
    std::fs::write(&filename, &serialized).unwrap();
}
