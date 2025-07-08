use crate::alignment::feature::FeaturePoint;
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for similarity metrics of [`Lorentzian`]s in
/// [`FeaturePoint`] form.
///
/// [`Lorentzian`]: crate::deconvolution::Lorentzian
pub(crate) trait Similarity {
    /// Computes the similarity between two [`FeaturePoint`]s.
    fn similarity(&self, first: &FeaturePoint, second: &FeaturePoint) -> f64;
}

/// Similarity metric for comparing two [`Lorentzian`]s.
///
/// [`Lorentzian`]: crate::deconvolution::Lorentzian
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum SimilarityMetric {
    /// Shape of the [`Lorentzian`]s.
    ///
    /// [`Lorentzian`]: crate::deconvolution::Lorentzian
    ///
    /// Uses the maximum and half-width of the [`Lorentzian`]s to assess their
    /// similarity.
    ///
    /// [`Lorentzian`]: crate::deconvolution::Lorentzian
    #[default]
    Shape,
    /// Shape and distance of the [`Lorentzian`]s.
    ///
    /// [`Lorentzian`]: crate::deconvolution::Lorentzian
    ///
    /// Uses the maximum, half-width and distance between the maxima of the
    /// [`Lorentzian`]s to assess their similarity.
    ///
    /// [`Lorentzian`]: crate::deconvolution::Lorentzian
    ShapeDistance,
}

impl std::fmt::Display for SimilarityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimilarityMetric::Shape => write!(f, "Shape"),
            SimilarityMetric::ShapeDistance => write!(f, "Shape Distance"),
        }
    }
}

impl Settings for SimilarityMetric {
    fn validate(&self) -> Result<()> {
        match self {
            SimilarityMetric::Shape => {}
            SimilarityMetric::ShapeDistance => {}
        }

        Ok(())
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        #[allow(clippy::match_like_matches_macro)]
        match (self, other) {
            (SimilarityMetric::Shape, SimilarityMetric::Shape) => true,
            (SimilarityMetric::ShapeDistance, SimilarityMetric::ShapeDistance) => true,
            _ => false,
        }
    }
}

/// Similarity metric based on the shape of two feature points.
#[derive(Debug, Default)]
pub(crate) struct ShapeSimilarity;

impl Similarity for ShapeSimilarity {
    fn similarity(&self, first: &FeaturePoint, second: &FeaturePoint) -> f64 {
        let max_term = f64::min(first.max(), second.max()) / f64::max(first.max(), second.max());
        let half_width_term = f64::min(first.hw(), second.hw()) / f64::max(first.hw(), second.hw());

        max_term * half_width_term
    }
}

impl ShapeSimilarity {
    /// Creates a new `ShapeSimilarity`.
    pub(crate) fn new() -> Self {
        Self
    }
}

/// Similarity metric based on the shape distance between two feature points.
///
/// Maximum distance is used to normalize the distance between the maxima of the
/// two [`Lorentzian`]s.
///
/// [`Lorentzian`]: crate::deconvolution::Lorentzian
#[derive(Debug)]
pub(crate) struct ShapeDistanceSimilarity {
    /// Maximum distance between two feature points.
    max_distance: f64,
}

impl Similarity for ShapeDistanceSimilarity {
    fn similarity(&self, first: &FeaturePoint, second: &FeaturePoint) -> f64 {
        let max_term = f64::min(first.max(), second.max()) / f64::max(first.max(), second.max());
        let half_width_term = f64::min(first.hw(), second.hw()) / f64::max(first.hw(), second.hw());
        let distance_term = 1.0 - f64::abs(first.maxp() - second.maxp()) / self.max_distance;

        max_term * half_width_term * distance_term
    }
}

impl ShapeDistanceSimilarity {
    /// Creates a new `ShapeDistanceSimilarity` with the given maximum distance.
    pub(crate) fn new(max_distance: f64) -> Self {
        Self { max_distance }
    }
}
