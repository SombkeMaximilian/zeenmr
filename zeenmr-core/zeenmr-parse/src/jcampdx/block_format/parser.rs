use crate::error::ByteRange;
use crate::jcampdx::block_format::error::{Error, Result};
use crate::jcampdx::block_format::{BlockFormat, BlockFormatBuilder, FormatToken};
use crate::jcampdx::parser::ChildParserExit;
use logos::{Lexer, Logos};

/// State of the [`FormatParser`].
///
/// Tracks which part of the sequence is being parsed currently.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum State {
    /// Encountered identifiers are before a [`Repeat`] token.
    ///
    /// [`Repeat`]: FormatToken::Repeat
    Prefix,
    /// Encountered identifiers are after an [`Increment`] token but before any
    /// [`Repeat`] token.
    ///
    /// [`Increment`]: FormatToken::Increment
    /// [`Repeat`]: FormatToken::Repeat
    AfterIncrement,
    /// Encountered identifiers are after a [`Repeat`] token.
    ///
    /// [`Repeat`]: FormatToken::Repeat
    Suffix,
}

/// Parser for the data block format specifiers of JCAMP-DX datablocks.
#[derive(Debug)]
pub(crate) struct FormatParser<'source> {
    /// Lexer for the data block format specifiers.
    lexer: Lexer<'source, FormatToken>,
    /// Start range for potential error reporting.
    start: ByteRange,
    /// Exit status of the parser (end of input or newline).
    exit: ChildParserExit,
    /// State of the parser.
    state: State,
    /// `BlockFormat` being constructed.
    builder: BlockFormatBuilder<'source>,
}

impl<'source> From<&'source str> for FormatParser<'source> {
    fn from(value: &'source str) -> Self {
        let lexer = FormatToken::lexer(value);
        let start = lexer.span().into();

        Self {
            lexer,
            start,
            state: State::Prefix,
            exit: ChildParserExit::default(),
            builder: BlockFormatBuilder::default(),
        }
    }
}

impl<'source, T> From<Lexer<'source, T>> for FormatParser<'source>
where
    T: Logos<'source, Source = str, Extras = ()>,
{
    fn from(value: Lexer<'source, T>) -> Self {
        let lexer = value.morph();
        let start = lexer.span().into();

        Self {
            lexer,
            start,
            state: State::Prefix,
            exit: ChildParserExit::default(),
            builder: BlockFormatBuilder::default(),
        }
    }
}

impl<'source> FormatParser<'source> {
    /// Recovers the [`Lexer`] from the `FormatParser`.
    pub(crate) fn into_lexer(self) -> Lexer<'source, FormatToken> {
        self.lexer
    }

    /// Parses the data block format specifier.
    ///
    /// The caller must ensure that the [`Lexer`] is at the start of a block
    /// format specifier string.
    ///
    /// # Errors
    ///
    /// Returns an error if the format specifier contains errors or the end of
    /// the input is reached during parsing. Unless the end of the input is
    /// reached, the [`Lexer`] will be synchronized to the end of the current
    /// line.
    pub(crate) fn parse_format(&mut self) -> Result<BlockFormat<'source>> {
        if let Err(e) = self.parse_tokens() {
            // if this fails the other error is unrecoverable anyway
            self.synchronize_end_of_line()?;
            return Err(e);
        }

        match self.exit {
            ChildParserExit::EndOfInput => Err(Error::end_of_input(self.lexer.span().into())),
            ChildParserExit::EndToken => self.finalize(),
        }
    }

    /// Main loop for parsing tokens.
    ///
    /// Advances the [`Lexer`] until an [`End`] token is encountered or the
    /// input ends.
    ///
    /// [`End`]: FormatToken::End
    fn parse_tokens(&mut self) -> Result<()> {
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                FormatToken::Identifier => self.identifier()?,
                FormatToken::Increment => self.increment()?,
                FormatToken::Repeat => self.repeat()?,
                FormatToken::DataBlockKind => {
                    self.data_block_kind()?;
                    self.exit = ChildParserExit::EndToken;
                    break;
                }
                FormatToken::End => {
                    self.exit = ChildParserExit::EndToken;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Advances the [`Lexer`] until the next [`End`] token is encountered.
    ///
    /// [`End`]: FormatToken::End
    ///
    /// This method is used for recovering from errors in a way that allows the
    /// higher-level parser to continue even after parsing the format specifier
    /// failed.
    ///
    /// # Errors
    ///
    /// Returns an error if the source ends before an [`End`] token is
    /// encountered
    ///
    /// [`End`]: FormatToken::End
    fn synchronize_end_of_line(&mut self) -> Result<()> {
        while let Some(token) = self.lexer.next().transpose()? {
            if token == FormatToken::End {
                self.exit = ChildParserExit::EndToken;
                break;
            }
        }

        match self.exit {
            ChildParserExit::EndOfInput => Err(Error::end_of_input(self.lexer.span().into())),
            ChildParserExit::EndToken => Ok(()),
        }
    }

    /// Finalizes the `BlockFormat`.
    ///
    /// # Errors
    ///
    /// Returns an error if the format specifier was empty, or if the prefix
    /// was not fully validated despite there being a suffix.
    fn finalize(&mut self) -> Result<BlockFormat<'source>> {
        let incrementing_set = self.builder.incrementing_is_some();

        if self.state == State::Suffix && !self.builder.prefix_was_validated() {
            // tracking the positions of increment and repeat would complicate
            // the parsing logic for the happy path. reconstructing them here by
            // looking for the repeat and then spanning until the terminating
            // tokens is cheap enough for the error path.
            let mut suffix = FormatToken::lexer(&self.lexer.source()[self.start.start..])
                .spanned()
                .filter_map(|(token, span)| token.ok().map(|token| (token, span)))
                .skip_while(|(token, _)| !matches!(token, FormatToken::Repeat));
            let start = suffix
                .next()
                .map(|(_, span)| self.start.start + span.end)
                .unwrap_or(self.start.start);
            let end = suffix
                .take_while(|(token, _)| matches!(token, FormatToken::Identifier))
                .last()
                .map(|(_, span)| self.start.start + span.end)
                .unwrap_or(self.start.end);

            return Err(Error::mismatched_repeat((start..end).into()));
        }

        match std::mem::take(&mut self.builder).finalize() {
            Some(block) => Ok(block),
            None => match self.state {
                State::Prefix => Err(Error::empty_format(self.start)),
                State::AfterIncrement if !incrementing_set => {
                    Err(Error::empty_increment(self.start))
                }
                State::AfterIncrement => {
                    let mut tokens = FormatToken::lexer(&self.lexer.source()[self.start.start..])
                        .spanned()
                        .take(2);
                    let start = tokens
                        .next()
                        .map(|(_, span)| span.start)
                        .unwrap_or(self.start.start);
                    let end = tokens
                        .next()
                        .map(|(_, span)| span.end)
                        .unwrap_or(self.start.end);

                    Err(Error::missing_repeat((start..end).into()))
                }
                State::Suffix => Err(Error::empty_repeat(self.start)),
            },
        }
    }

    /// Handles [`Identifier`] tokens.
    ///
    /// [`Identifier`]: FormatToken::Identifier
    ///
    /// [`Identifier`]s before a [`Repeat`] token are inserted into the prefix,
    /// while the suffix, i.e., after a [`Repeat`], are compared against the
    /// prefix one by one to ensure identity.
    ///
    /// [`Identifier`]: FormatToken::Identifier
    /// [`Repeat`]: FormatToken::Repeat
    fn identifier(&mut self) -> Result<()> {
        match self.state {
            State::Prefix | State::AfterIncrement => {
                self.builder.push(self.lexer.slice());

                Ok(())
            }
            State::Suffix => {
                if !self.builder.compare_prefix(self.lexer.slice()) {
                    Err(Error::mismatched_repeat(self.lexer.span().into()))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Handles [`Increment`] tokens.
    ///
    /// [`Increment`]: FormatToken::Increment
    ///
    /// Switches into the [`AfterIncrement`] state and moves the prefix into the
    /// increment variable.
    ///
    /// [`AfterIncrement`]: State::AfterIncrement
    ///
    /// # Errors
    ///
    /// Returns an error if the prefix isn't exactly one identifier, or if the
    /// parser is not in the [`Prefix`] state.
    ///
    /// [`Prefix`]: State::Prefix
    fn increment(&mut self) -> Result<()> {
        match self.state {
            State::Prefix => match self.builder.len() {
                0 => Err(Error::empty_increment(self.lexer.span().into())),
                1 => {
                    self.state = State::AfterIncrement;
                    let incrementing = self
                        .builder
                        .pop()
                        .expect("prefix len must be 1 to reach this");
                    self.builder.set_incrementing(incrementing);

                    Ok(())
                }
                _ => Err(Error::multiple_identifier_increment(
                    self.lexer.span().into(),
                )),
            },
            State::AfterIncrement => Err(Error::multiple_increment(self.lexer.span().into())),
            State::Suffix => Err(Error::increment_after_repeat(self.lexer.span().into())),
        }
    }

    /// Handles [`Repeat`] tokens.
    ///
    /// [`Repeat`]: FormatToken::Repeat
    ///
    /// Switches into the [`Suffix`] state.
    ///
    /// [`Suffix`]: State::Suffix
    ///
    /// # Errors
    ///
    /// Returns an error if the prefix is empty, or if the parser is already in
    /// the [`Suffix`] state.
    ///
    /// [`Suffix`]: State::Suffix
    fn repeat(&mut self) -> Result<()> {
        match self.state {
            State::Prefix | State::AfterIncrement => {
                if self.builder.is_empty() {
                    Err(Error::empty_repeat(self.lexer.span().into()))
                } else {
                    self.state = State::Suffix;

                    Ok(())
                }
            }
            State::Suffix => Err(Error::multiple_repeat(self.lexer.span().into())),
        }
    }

    /// Handles [`DataBlockKind`] tokens and checks for a subsequent [`End`]
    /// token.
    ///
    /// [`DataBlockKind`]: FormatToken::DataBlockKind
    /// [`End`]: FormatToken::End
    ///
    /// # Errors
    ///
    /// Returns an error if the input ends after the [`DataBlockKind`] token.
    /// Note that due to how this token is matched, the input will always end
    /// after it, unless it's followed by a newline character, and therefore an
    /// [`End`] token.
    ///
    /// [`DataBlockKind`]: FormatToken::DataBlockKind
    /// [`End`]: FormatToken::End
    fn data_block_kind(&mut self) -> Result<()> {
        let block_kind = self.lexer.slice()[1..].trim();
        self.builder.set_block_kind(block_kind);

        if self.lexer.next().transpose()?.is_some() {
            Ok(())
        } else {
            Err(Error::end_of_input(self.lexer.span().into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jcampdx::block_format::LineLayout;

    macro_rules! parser_test {
        ($name:ident, $data:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $expected;
                let mut parser = FormatParser::from(data);
                let parsed = parser.parse_format();

                assert_eq!(parsed, expected);
            }
        };
    }

    parser_test!(
        repeating,
        "(X++(Y..Y))\n",
        Ok(BlockFormat {
            line_layout: LineLayout::RepeatingValue {
                incrementing: "X",
                repeating: "Y",
            },
            kind: None,
        })
    );
    parser_test!(
        repeating_block_kind,
        "(X++(R..R)), XYDATA\n",
        Ok(BlockFormat {
            line_layout: LineLayout::RepeatingValue {
                incrementing: "X",
                repeating: "R",
            },
            kind: Some("XYDATA"),
        })
    );
    parser_test!(
        multi_group,
        "(XY..XY)\n",
        Ok(BlockFormat {
            line_layout: LineLayout::GroupedValues(vec!["X", "Y"]),
            kind: None,
        })
    );
    parser_test!(
        multi_group_block_kind,
        "(XY..XY), PEAKS\n",
        Ok(BlockFormat {
            line_layout: LineLayout::GroupedValues(vec!["X", "Y"]),
            kind: Some("PEAKS"),
        })
    );
    parser_test!(
        single_group,
        "(XYWA)\n",
        Ok(BlockFormat {
            line_layout: LineLayout::GroupedValues(vec!["X", "Y", "W", "A",]),
            kind: None,
        })
    );
    parser_test!(
        single_group_block_kind,
        "(XYWA), PEAK ASSIGNMENTS\n",
        Ok(BlockFormat {
            line_layout: LineLayout::GroupedValues(vec!["X", "Y", "W", "A",]),
            kind: Some("PEAK ASSIGNMENTS"),
        })
    );
    parser_test!(
        invalid_literal,
        "(X.Y)\n",
        Err(Error::invalid_literal(ByteRange::new(2, 3)))
    );
    parser_test!(
        end_of_input,
        "(X..Y)",
        Err(Error::end_of_input(ByteRange::new(6, 6)))
    );
    parser_test!(
        empty_format,
        "\n",
        Err(Error::empty_format(ByteRange::new(0, 0)))
    );
    parser_test!(
        empty_repeat,
        "X++(..Y)\n",
        Err(Error::empty_repeat(ByteRange::new(4, 6)))
    );
    parser_test!(
        missing_repeat,
        "X++\n",
        Err(Error::missing_repeat(ByteRange::new(0, 3)))
    );
    parser_test!(
        mismatched_repeat_identifiers,
        "X++(Y..Z)\n",
        Err(Error::mismatched_repeat(ByteRange::new(7, 8)))
    );
    parser_test!(
        mismatched_repeat_length_longer,
        "(XY..XYZ)\n",
        Err(Error::mismatched_repeat(ByteRange::new(7, 8)))
    );
    parser_test!(
        mismatched_repeat_length_shorter,
        "(XYZ..XY)\n",
        Err(Error::mismatched_repeat(ByteRange::new(6, 8)))
    );
    parser_test!(
        multiple_increment,
        "(X++X++(Y..Y))\n",
        Err(Error::multiple_increment(ByteRange::new(5, 7)))
    );
    parser_test!(
        multiple_repeat,
        "(X++(Y..Y..Y)\n",
        Err(Error::multiple_repeat(ByteRange::new(9, 11)))
    );
    parser_test!(
        increment_after_repeat,
        "(Y..Y)++\n",
        Err(Error::increment_after_repeat(ByteRange::new(6, 8)))
    );
    parser_test!(
        multiple_identifier_increment,
        "XF1++(Y..Y)\n",
        Err(Error::multiple_identifier_increment(ByteRange::new(3, 5)))
    );
}
