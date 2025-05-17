use crate::alignment::assignment::Filter;
use crate::alignment::feature::{FeatureLayer, FeatureMap};
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for selecting pairs of feature maps for alignment.
pub(crate) trait Strategy: Send + Sync + std::fmt::Debug {
    /// Selects pairs of feature maps for alignment based on the given
    /// deconvolutions.
    fn generate_maps(&self, layers: &[FeatureLayer], filter: &dyn Filter) -> Vec<FeatureMap>;

    /// Returns the settings of the trait object.
    fn settings(&self) -> AlignmentStrategy;
}

/// Alignment strategy for deconvolutions.
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum AlignmentStrategy {
    /// Aligns all [`Deconvolution`]s to the reference [`Deconvolution`].
    ///
    /// [`Deconvolution`]: crate::deconvolution::Deconvolution
    Reference(usize),
    /// Aligns all [`Deconvolution`]s in a pairwise fashion.
    ///
    /// [`Deconvolution`]: crate::deconvolution::Deconvolution
    Pairwise,
}

impl Default for AlignmentStrategy {
    fn default() -> Self {
        AlignmentStrategy::Reference(0)
    }
}

impl std::fmt::Display for AlignmentStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlignmentStrategy::Reference(index) => {
                write!(f, "Reference Alignment [index: {index}]")
            }
            AlignmentStrategy::Pairwise => write!(f, "Pairwise Alignment"),
        }
    }
}

impl Settings for AlignmentStrategy {
    fn validate(&self) -> Result<()> {
        match self {
            AlignmentStrategy::Reference(..) => {}
            AlignmentStrategy::Pairwise => {}
        }

        Ok(())
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (AlignmentStrategy::Reference(index1), AlignmentStrategy::Reference(index2)) => {
                index1 == index2
            }
            (AlignmentStrategy::Pairwise, AlignmentStrategy::Pairwise) => true,
            _ => false,
        }
    }
}
