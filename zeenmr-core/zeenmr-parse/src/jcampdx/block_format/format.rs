/// Layout of the lines in a data block.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) enum LineLayout<'source> {
    /// `XYDATA` specific layout.
    ///
    /// Each line contains one value for the first identifier, typically `X`,
    /// and then repeats values for the second identifier, typically `Y`, `R` or
    /// `I`, until the line ends.
    RepeatingValue {
        incrementing: &'source str,
        repeating: &'source str,
    },
    /// Grouped values enclosed by parentheses, or separated by semicolons or
    /// newlines.
    ///
    /// Each line contains groups of values for the identifiers until the line
    /// ends. Typically, groups are not meant to extend beyond a linebreak, but
    /// they may.
    GroupedValues(Vec<&'source str>),
}

/// Format of a data block.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct BlockFormat<'source> {
    /// Layout of the lines.
    pub(crate) line_layout: LineLayout<'source>,
    /// Optional kind descriptor.
    pub(crate) kind: Option<&'source str>,
}

/// Builder pattern for [`BlockFormat`].
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct BlockFormatBuilder<'source> {
    /// Main identifiers.
    identifiers: Vec<&'source str>,
    /// Current identifier to check against, may be out of bounds.
    suffix_check: usize,
    /// Incrementing variable, if any.
    incrementing: Option<&'source str>,
    /// Block kind of `DATA TABLE` data block.
    block_kind: Option<&'source str>,
}

impl<'source> BlockFormatBuilder<'source> {
    /// Finalizes the `BlockFormat`, or returns `None` if one cannot be
    /// constructed.
    ///
    /// # Layout Requirements
    ///
    /// - [`RepeatingValue`]: `identifiers` contains exactly one name and
    ///   `incrementing` is set.
    /// - [`GroupedValues`]: `identifiers` is non-empty.
    ///
    /// [`RepeatingValue`]: LineLayout::RepeatingValue
    /// [`GroupedValues`]: LineLayout::GroupedValues
    pub(crate) fn finalize(self) -> Option<BlockFormat<'source>> {
        if self.identifiers.is_empty() {
            return None;
        }

        match self.incrementing {
            Some(incrementing) if self.identifiers.len() == 1 => Some(BlockFormat {
                line_layout: LineLayout::RepeatingValue {
                    incrementing,
                    repeating: self.identifiers[0],
                },
                kind: self.block_kind,
            }),
            None if !self.identifiers.is_empty() => Some(BlockFormat {
                line_layout: LineLayout::GroupedValues(self.identifiers),
                kind: self.block_kind,
            }),
            _ => None,
        }
    }

    /// Returns `true` if the incrementing variable identifier was set.
    pub(crate) fn incrementing_is_some(&self) -> bool {
        self.incrementing.is_some()
    }

    /// Returns `true` if at least one identifier was pushed and
    /// [`compare_prefix`] was called as many times as there are identifiers.
    pub(crate) fn prefix_was_validated(&self) -> bool {
        !self.is_empty() && self.suffix_check == self.identifiers.len()
    }

    /// Returns `true` if the identifiers stack contains no elements.
    pub(crate) fn is_empty(&self) -> bool {
        self.identifiers.is_empty()
    }

    /// Returns the number of elements in the identifiers stack.
    pub(crate) fn len(&self) -> usize {
        self.identifiers.len()
    }

    /// Returns `true` if suffix matches the currently checked identifier.
    ///
    /// Starts at the first identifier and each call of this method increments
    /// the check index. If the check index is out of bounds, `false` is
    /// returned.
    pub(crate) fn compare_prefix(&mut self, suffix: &'source str) -> bool {
        let result = matches!(
            self.identifiers.get(self.suffix_check),
            Some(prefix) if *prefix == suffix
        );
        self.suffix_check += 1;

        result
    }

    /// Sets the incrementing variable identifier.
    pub(crate) fn set_incrementing(&mut self, incrementing: &'source str) {
        self.incrementing = Some(incrementing);
    }

    /// Sets the data block kind.
    pub(crate) fn set_block_kind(&mut self, kind: &'source str) {
        self.block_kind = Some(kind);
    }

    /// Pushes an identifier onto the stack.
    pub(crate) fn push(&mut self, prefix: &'source str) {
        self.identifiers.push(prefix);
    }

    /// Removes the last element from the stack and returns it, or `None` if it
    /// is empty.
    pub(crate) fn pop(&mut self) -> Option<&'source str> {
        self.identifiers.pop()
    }
}
