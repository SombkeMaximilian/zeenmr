use metabodecon::alignment::{
    Aligner, AlignmentStrategy, FilteringSettings, SimilarityMetric, SolvingSettings,
};
use metabodecon::deconvolution::Deconvoluter;
use metabodecon::spectrum::Bruker;

#[test]
fn blood() {
    let spectra = Bruker::read_spectra("../data/bruker/blood", 10, 10, (-2.2, 11.8)).unwrap();
    let deconvoluter = Deconvoluter::default();
    let deconvolutions = deconvoluter
        .deconvolute_spectra(&spectra)
        .unwrap();
    let aligner = Aligner::new(
        AlignmentStrategy::Reference(0),
        FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: 0.5,
        },
        SolvingSettings::LinearProgramming,
    )
    .unwrap();
    let _ = aligner.align_deconvolutions(&deconvolutions);
}
