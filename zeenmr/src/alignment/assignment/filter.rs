use crate::alignment::assignment::{Assignment, SimilarityMetric};
use crate::alignment::error::{Error, Kind};
use crate::alignment::feature::FeatureLayer;
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for assignment candidate filtering.
pub(crate) trait Filter: Send + Sync + std::fmt::Debug {
    /// Generates all possible assignments between two feature layers and
    /// filters them.
    fn filter_assignments(&self, first: &FeatureLayer, second: &FeatureLayer) -> Vec<Assignment>;

    /// Returns the settings of the trait object.
    fn settings(&self) -> FilteringSettings;
}

/// Filtering settings for the assignment candidate selection process.
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum FilteringSettings {
    /// Maximum distance and minimum similarity between two [`Lorentzian`]s to
    /// be valid candidates.
    ///
    /// [`Lorentzian`]: crate::deconvolution::Lorentzian
    DistanceSimilarity {
        /// Similarity metric.
        similarity_metric: SimilarityMetric,
        /// Maximum distance.
        max_distance: f64,
        /// Minimum similarity.
        min_similarity: f64,
    },
}

impl Default for FilteringSettings {
    fn default() -> Self {
        Self::DistanceSimilarity {
            similarity_metric: SimilarityMetric::default(),
            max_distance: 0.025,
            min_similarity: 0.5,
        }
    }
}

impl std::fmt::Display for FilteringSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilteringSettings::DistanceSimilarity {
                similarity_metric,
                max_distance,
                min_similarity,
            } => {
                write!(
                    f,
                    "Distance Similarity Filter [\
                     similarity metric: {similarity_metric},\
                     maximum distance: {max_distance},\
                     minimum similarity: {min_similarity}\
                     ]"
                )
            }
        }
    }
}

impl Settings for FilteringSettings {
    fn validate(&self) -> Result<()> {
        match self {
            FilteringSettings::DistanceSimilarity {
                similarity_metric,
                max_distance,
                min_similarity,
            } => {
                similarity_metric.validate()?;
                if *max_distance < 0.0
                    || *min_similarity < 0.0
                    || *min_similarity > 1.0
                    || !max_distance.is_finite()
                    || !min_similarity.is_finite()
                {
                    return Err(
                        Error::new(Kind::InvalidFilteringSettings { settings: *self }).into(),
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (
                FilteringSettings::DistanceSimilarity {
                    similarity_metric: similarity_metric1,
                    max_distance: max_distance1,
                    min_similarity: min_similarity1,
                },
                FilteringSettings::DistanceSimilarity {
                    similarity_metric: similarity_metric2,
                    max_distance: max_distance2,
                    min_similarity: min_similarity2,
                },
            ) => {
                similarity_metric1.compare(similarity_metric2)
                    && float_cmp::approx_eq!(f64, *max_distance1, *max_distance2)
                    && float_cmp::approx_eq!(f64, *min_similarity1, *min_similarity2)
            }
        }
    }
}
