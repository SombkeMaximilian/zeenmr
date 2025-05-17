use crate::Result;

/// Trait interface for settings enums.
pub(crate) trait Settings: Default + Clone {
    /// Validates the settings.
    fn validate(&self) -> Result<()>;

    /// Compares the settings with another instance.
    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool;
}
