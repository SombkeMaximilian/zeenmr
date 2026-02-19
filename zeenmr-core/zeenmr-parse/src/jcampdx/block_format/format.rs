use crate::jcampdx::ChildParserExit;

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
pub(crate) struct BlockFormat {
    /// Exit status of the parser (end of input or newline).
    pub(crate) exit: ChildParserExit,
    /// Layout of the lines.
    pub(crate) line_layout: LineLayout,
    /// Optional kind descriptor.
    pub(crate) kind: Option<String>,
}

/// Builder pattern for [`BlockFormat`].
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct BlockFormatBuilder<'source> {
    /// Exit status of the parser (end of input or newline).
    exit: ChildParserExit,
    /// Prefix identifiers.
    prefix: Vec<&'source str>,
    /// Suffix identifiers.
    suffix: Vec<&'source str>,
    /// Incrementing variable, if any.
    incrementing: Option<&'source str>,
    /// Block kind of `DATA TABLE` data block.
    block_kind: Option<&'source str>,
}

impl<'source> BlockFormatBuilder<'source> {
    /// Finalizes the `BlockFormat` using a [`RepeatingValue`] line layout.
    ///
    /// [`RepeatingValue`]: LineLayout::RepeatingValue
    ///
    /// Returns `None` if the prefix length is not 1, or if the incrementing
    /// variable identifier is not set.
    pub(crate) fn finalize_repeating(self) -> Option<BlockFormat> {
        if self.prefix.len() == 1 {
            self.incrementing.map(|incrementing| BlockFormat {
                exit: self.exit,
                line_layout: LineLayout::RepeatingValue {
                    incrementing: incrementing.to_string(),
                    repeating: self.prefix[0].to_string(),
                },
                kind: self.block_kind.map(ToString::to_string),
            })
        } else {
            None
        }
    }

    /// Finalizes the `BlockFormat` using a [`SingleGroup`] line layout.
    ///
    /// [`SingleGroup`]: LineLayout::SingleGroup
    pub(crate) fn finalize_single_group(self) -> BlockFormat {
        let identifiers = self
            .prefix
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();

        BlockFormat {
            exit: self.exit,
            line_layout: LineLayout::SingleGroup(identifiers),
            kind: self.block_kind.map(ToString::to_string),
        }
    }

    /// Finalizes the `BlockFormat` using a [`MultiGroup`] line layout.
    ///
    /// [`MultiGroup()`]: LineLayout::MultiGroup
    pub(crate) fn finalize_multi_group(self) -> BlockFormat {
        let identifiers = self
            .suffix
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();

        BlockFormat {
            exit: self.exit,
            line_layout: LineLayout::MultiGroup(identifiers),
            kind: self.block_kind.map(ToString::to_string),
        }
    }

    /// Returns `true` if the exit status is set to [`EndToken`].
    ///
    /// [`EndToken`]: ChildParserExit::EndToken
    pub(crate) fn is_newline_exit(&self) -> bool {
        self.exit == ChildParserExit::EndToken
    }

    /// Returns `true` if the incrementing variable identifier was set.
    pub(crate) fn incrementing_is_some(&self) -> bool {
        self.incrementing.is_some()
    }

    /// Returns `true` if the block kind was set.
    pub(crate) fn block_kind_is_some(&self) -> bool {
        self.block_kind.is_some()
    }

    /// Returns `true` if the prefix contains no elements.
    pub(crate) fn prefix_is_empty(&self) -> bool {
        self.prefix.is_empty()
    }

    /// Returns the number of elements in the prefix stack.
    pub(crate) fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    /// Returns `true` if the prefix and suffix are equal.
    pub(crate) fn prefix_matches_suffix(&self) -> bool {
        self.prefix == self.suffix
    }

    /// Updates the exit status to [`EndToken`].
    ///
    /// [`EndToken`]: ChildParserExit::EndToken
    pub(crate) fn newline_exit(&mut self) {
        self.exit = ChildParserExit::EndToken;
    }

    /// Sets the incrementing variable identifier.
    pub(crate) fn set_incrementing(&mut self, incrementing: &'source str) {
        self.incrementing = Some(incrementing);
    }

    /// Sets the data block kind.
    pub(crate) fn set_block_kind(&mut self, kind: &'source str) {
        self.block_kind = Some(kind);
    }

    /// Pushes a prefix identifier onto the stack.
    pub(crate) fn push_prefix(&mut self, prefix: &'source str) {
        self.prefix.push(prefix);
    }

    /// Pushes a suffix identifier onto the stack.
    pub(crate) fn push_suffix(&mut self, suffix: &'source str) {
        self.suffix.push(suffix);
    }

    /// Removes the last element from the prefix stack and returns it, or `None`
    /// if it is empty.
    pub(crate) fn pop_prefix(&mut self) -> Option<&'source str> {
        self.prefix.pop()
    }
}
