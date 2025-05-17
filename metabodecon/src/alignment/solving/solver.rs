use crate::alignment::feature::FeatureMap;
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for solving the assignment problem.
pub(crate) trait Solver: Send + Sync + std::fmt::Debug {
    /// Solves the assignment problem for the given feature maps.
    fn solve(&self, candidate_maps: Vec<FeatureMap>) -> Vec<FeatureMap>;

    /// Returns the settings of the trait object.
    fn settings(&self) -> SolvingSettings;
}

/// Solving settings for the assignment problem.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum SolvingSettings {
    /// Formulates the assignment problem in a linear programming framework.
    ///
    /// Uses the [good_lp] crate with the [HiGHS] solver to solve the assignment
    /// problem.
    ///
    /// [good_lp]: https://docs.rs/good_lp/
    /// [HiGHS]: https://highs.dev/
    #[default]
    LinearProgramming,
}

impl std::fmt::Display for SolvingSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolvingSettings::LinearProgramming => write!(f, "Linear Programming"),
        }
    }
}

impl Settings for SolvingSettings {
    fn validate(&self) -> Result<()> {
        match self {
            SolvingSettings::LinearProgramming => {}
        }

        Ok(())
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (SolvingSettings::LinearProgramming, SolvingSettings::LinearProgramming) => true,
        }
    }
}
