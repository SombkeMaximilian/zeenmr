/// Layout of the lines in a data block.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) enum LineLayout {
    /// `XYDATA` specific layout.
    ///
    /// Each line contains one value for the first identifier, typically `X`,
    /// and then repeats values for the second identifier, typically `Y`, `R` or
    /// `I`, until the line ends.
    RepeatingValue {
        incrementing: String,
        repeating: String,
    },
    /// Grouped values enclosed by parentheses or separated by semicolons.
    ///
    /// Each line contains groups of values for the identifiers until the line
    /// ends. Typically, groups are not meant to extend beyond a linebreak, but
    /// they may.
    MultiGroup(Vec<String>),
    /// Grouped values separated by line breaks.
    ///
    /// Each line contains one group of values for the identifiers.
    SingleGroup(Vec<String>),
}

/// Format of a data block.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct BlockFormat<'source> {
    /// Layout of the lines.
    pub(crate) line_layout: LineLayout,
    /// Optional kind descriptor.
    pub(crate) kind: Option<&'source str>,
}

impl<'source> BlockFormat<'source> {
    pub(crate) fn new(line_layout: LineLayout, kind: Option<&'source str>) -> Self {
        Self { line_layout, kind }
    }
}
