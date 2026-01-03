use crate::jcampdx::block_format::Identifier;

#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) enum BlockFormat {
    /// `XYDATA` specific layout.
    ///
    /// Each line contains one value for the first identifier, typically `X`,
    /// and then repeats values for the second identifier, typically `Y`, `R` or
    /// `I`, until the line ends.
    RepeatingValue {
        increment: Identifier,
        repeating: Identifier,
    },
    /// Grouped values enclosed by parentheses or separated by semicolons.
    ///
    /// Each line contains groups of values for the identifiers until the line
    /// ends. Typically, groups are not meant to extend beyond a linebreak, but
    /// they may.
    MultiGroup(Vec<Identifier>),
    /// Grouped values separated by line breaks.
    ///
    /// Each line contains one group of values for the identifiers.
    SingleGroup(Vec<Identifier>),
}
