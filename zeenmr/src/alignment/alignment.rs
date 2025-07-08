use crate::alignment::{AlignmentStrategy, FilteringSettings, SolvingSettings};
use crate::deconvolution::Deconvolution;
use std::sync::Arc;

#[cfg(feature = "serde")]
use crate::alignment::SerializedAlignment;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data structure representing an alignment of multiple deconvolutions.
///
/// An `Alignment` contains a collection of [`Deconvolution`]s which have their
/// deconvoluted signals aligned.
///
/// Constructing an `Alignment` directly is typically not the intended use case.
/// Instead, it is the output of the alignment functions provided by the
/// [`Aligner`] type.
///
/// [`Aligner`]: crate::alignment::Aligner
///
/// # Thread Safety
///
/// The `Alignment` type is both [`Send`] and [`Sync`], allowing it to be
/// safely shared and accessed across threads.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(into = "SerializedAlignment", try_from = "SerializedAlignment")
)]
pub struct Alignment {
    /// Aligned `Deconvolution`s.
    deconvolutions: Arc<[Deconvolution]>,
    /// Strategy used for aligning the deconvolutions.
    alignment_strategy: AlignmentStrategy,
    /// Settings for filtering the feature maps.
    filtering_settings: FilteringSettings,
    /// Settings for solving the assignment problem.
    solving_settings: SolvingSettings,
}

impl AsRef<Alignment> for Alignment {
    fn as_ref(&self) -> &Alignment {
        self
    }
}

impl Alignment {
    /// Constructs a new `Alignment`.
    ///
    /// Normally, this type is only instantiated by the alignment functions
    /// provided by the [`Aligner`] type.
    ///
    /// [`Aligner`]: crate::alignment::Aligner
    pub fn new<I: IntoIterator<Item = Deconvolution>>(
        deconvolutions: I,
        alignment_strategy: AlignmentStrategy,
        filtering_settings: FilteringSettings,
        solving_settings: SolvingSettings,
    ) -> Self {
        Self {
            deconvolutions: deconvolutions.into_iter().collect(),
            alignment_strategy,
            filtering_settings,
            solving_settings,
        }
    }

    /// Returns the aligned deconvolutions as a slice of [`Deconvolution`].
    pub fn deconvolutions(&self) -> &[Deconvolution] {
        &self.deconvolutions
    }

    /// Returns the alignment strategy used for this alignment.
    pub fn alignment_strategy(&self) -> AlignmentStrategy {
        self.alignment_strategy
    }

    /// Returns the filtering settings used for this alignment.
    pub fn filtering_settings(&self) -> FilteringSettings {
        self.filtering_settings
    }

    /// Returns the solving settings used for this alignment.
    pub fn solving_settings(&self) -> SolvingSettings {
        self.solving_settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(Alignment);
        assert_sync!(Alignment);
    }
}
