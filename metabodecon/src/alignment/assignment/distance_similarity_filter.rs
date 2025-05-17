use crate::alignment::FilteringSettings;
use crate::alignment::assignment::{
    Assignment, Filter, ShapeDistanceSimilarity, ShapeSimilarity, Similarity, SimilarityMetric,
};
use crate::alignment::feature::FeatureLayer;

/// Generates all possible assignments between two feature layers and filters
/// out ones that don't pass the filter.
///
/// This filter uses the similarity and distance between two [`Lorentzian`]s to
/// assess whether they are valid candidates for alignment. Each assignment is
/// given a score based on the similarity metric. The filter will only keep
/// assignments that have a distance less than or equal to the maximum distance
/// and a similarity greater than or equal to the minimum similarity.
///
/// [`Lorentzian`]: crate::deconvolution::Lorentzian
///
/// ````text
/// similarity >= min_similarity
/// distance <= max_distance
/// ```
#[derive(Debug)]
pub(crate) struct DistanceSimilarityFilter {
    similarity_metric: SimilarityMetric,
    max_distance: f64,
    min_similarity: f64,
}

impl Filter for DistanceSimilarityFilter {
    fn filter_assignments(&self, first: &FeatureLayer, second: &FeatureLayer) -> Vec<Assignment> {
        let similarity: Box<dyn Similarity> = match self.similarity_metric {
            SimilarityMetric::Shape => Box::new(ShapeSimilarity::new()),
            SimilarityMetric::ShapeDistance => {
                Box::new(ShapeDistanceSimilarity::new(self.max_distance))
            }
        };

        first
            .iter()
            .enumerate()
            .flat_map(|(i, first_feature)| {
                second
                    .iter()
                    .enumerate()
                    .filter_map(|(j, second_feature)| {
                        let distance = f64::abs(first_feature.maxp() - second_feature.maxp());
                        let similarity = similarity.similarity(first_feature, second_feature);

                        if distance <= self.max_distance && similarity >= self.min_similarity {
                            Some(Assignment::new(i, j, similarity))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn settings(&self) -> FilteringSettings {
        FilteringSettings::DistanceSimilarity {
            similarity_metric: self.similarity_metric,
            max_distance: self.max_distance,
            min_similarity: self.min_similarity,
        }
    }
}

impl DistanceSimilarityFilter {
    /// Creates a new `DistanceSimilarityFilter` with the given maximum distance
    /// and minimum similarity.
    pub(crate) fn new(
        similarity_metric: SimilarityMetric,
        max_distance: f64,
        min_similarity: f64,
    ) -> Self {
        Self {
            similarity_metric,
            max_distance,
            min_similarity,
        }
    }
}
