use crate::jcampdx::block_format::error::{Error, Result};
use crate::jcampdx::block_format::{BlockFormat, FormatToken, LineLayout};
use crate::{Cursor, Location, Position};
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

/// Exit status of the [`FormatParser`].
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
enum ExitStatus {
    /// Parsing was terminated by the end of the input.
    #[default]
    EndOfInput,
    /// Parsing was terminated by encountering a newline.
    EndToken,
}

/// Parser for the data block format specifiers of JCAMP-DX datablocks.
#[derive(Debug)]
pub(crate) struct FormatParser<'source> {
    /// Lexer for the data block format specifiers.
    lexer: Lexer<'source, FormatToken>,
    /// State of the parser.
    state: State,
    /// Exit status (newline or end of the source encountered).
    exit: ExitStatus,
    /// Start position for potential error reporting.
    start: Position,
    /// Prefix identifiers.
    prefix: Vec<&'source str>,
    /// Suffix identifiers.
    suffix: Vec<&'source str>,
    /// Incrementing variable, if any.
    increment: Vec<&'source str>,
    /// Block kind of `DATA TABLE` data block.
    block_kind: Option<&'source str>,
}

impl<'source> From<&'source str> for FormatParser<'source> {
    fn from(value: &'source str) -> Self {
        let lexer = FormatToken::lexer(value);
        let start = lexer.location();

        Self {
            lexer,
            start,
            state: State::Prefix,
            prefix: Vec::new(),
            suffix: Vec::new(),
            increment: Vec::new(),
            block_kind: None,
            exit: ExitStatus::default(),
        }
    }
}

impl<'source, T> From<Lexer<'source, T>> for FormatParser<'source>
where
    T: Logos<'source, Source = str> + Clone,
    T::Extras: Clone + Into<Cursor>,
{
    fn from(value: Lexer<'source, T>) -> Self {
        let lexer = value.morph();
        let start = lexer.location();

        Self {
            lexer,
            start,
            state: State::Prefix,
            prefix: Vec::new(),
            suffix: Vec::new(),
            increment: Vec::new(),
            block_kind: None,
            exit: ExitStatus::EndOfInput,
        }
    }
}

impl<'source> FormatParser<'source> {
    /// Recovers the [`Lexer`] from the `FormatParser`.
    pub(crate) fn into_lexer(mut self) -> Lexer<'source, FormatToken> {
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
    pub(crate) fn parse_format(&mut self) -> Result<BlockFormat> {
        if let Err(e) = self.parse_tokens() {
            // if this fails the other error is unrecoverable anyway
            self.synchronize_end_of_line()?;
            return Err(e);
        }

        match self.exit {
            ExitStatus::EndToken => self.finalize(),
            ExitStatus::EndOfInput => Err(Error::end_of_input(self.lexer.location())),
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
                FormatToken::Identifier => self.identifier(),
                FormatToken::Increment => self.increment()?,
                FormatToken::Repeat => self.repeat()?,
                FormatToken::DataBlockKind => {
                    self.data_block_kind()?;
                    self.exit = ExitStatus::EndToken;
                    break;
                }
                FormatToken::End => {
                    self.exit = ExitStatus::EndToken;
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
                self.exit = ExitStatus::EndToken;
                break;
            }
        }

        match self.exit {
            ExitStatus::EndToken => Ok(()),
            ExitStatus::EndOfInput => Err(Error::end_of_input(self.lexer.location())),
        }
    }

    fn finalize(&mut self) -> Result<BlockFormat> {
        match self.state {
            State::Prefix => {
                let identifiers = self
                    .prefix
                    .iter()
                    .map(|identifier| (*identifier).into())
                    .collect();

                Ok(BlockFormat::new(
                    LineLayout::SingleGroup(identifiers),
                    self.block_kind,
                ))
            }
            State::AfterIncrement => {
                let increment = self.increment[0].into();
                let repeating = self.prefix[0].into();

                Ok(BlockFormat::new(
                    LineLayout::RepeatingValue {
                        incrementing: increment,
                        repeating,
                    },
                    self.block_kind,
                ))
            }
            State::Suffix => {
                if self.prefix != self.suffix {
                    return Err(Error::mismatched_repeat(self.start));
                }

                if self.increment.is_empty() {
                    let identifiers = self
                        .suffix
                        .iter()
                        .map(|identifier| (*identifier).into())
                        .collect();

                    Ok(BlockFormat::new(
                        LineLayout::MultiGroup(identifiers),
                        self.block_kind,
                    ))
                } else {
                    let increment = self.increment[0].into();
                    let repeating = self.prefix[0].into();

                    Ok(BlockFormat::new(
                        LineLayout::RepeatingValue {
                            incrementing: increment,
                            repeating,
                        },
                        self.block_kind,
                    ))
                }
            }
        }
    }

    /// Handles [`Identifier`] tokens.
    ///
    /// [`Identifier`]: FormatToken::Identifier
    ///
    /// [`Identifier`]s before a [`Repeat`] token are inserted into the prefix,
    /// while those after a [`Repeat`] are inserted into the suffix.
    ///
    /// [`Identifier`]: FormatToken::Identifier
    /// [`Repeat`]: FormatToken::Repeat
    fn identifier(&mut self) {
        match self.state {
            State::Prefix | State::AfterIncrement => self.prefix.push(self.lexer.slice()),
            State::Suffix => self.suffix.push(self.lexer.slice()),
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
            State::Prefix => {
                if self.prefix.len() == 1 {
                    self.state = State::AfterIncrement;
                    std::mem::swap(&mut self.prefix, &mut self.increment);

                    Ok(())
                } else {
                    Err(Error::multiple_identifier_increment(self.lexer.location()))
                }
            }
            State::AfterIncrement => Err(Error::multiple_increment(self.lexer.location())),
            State::Suffix => Err(Error::increment_after_repeat(self.lexer.location())),
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
                if self.prefix.is_empty() {
                    Err(Error::empty_repeat(self.lexer.location()))
                } else {
                    self.state = State::Suffix;

                    Ok(())
                }
            }
            State::Suffix => Err(Error::multiple_repeat(self.lexer.location())),
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
        match self.block_kind {
            Some(_) => unreachable!(),
            None => {
                self.block_kind = Some(&self.lexer.slice()[1..].trim());

                if self.lexer.next().transpose()?.is_some() {
                    Ok(())
                } else {
                    Err(Error::end_of_input(self.lexer.location()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! parser_test {
        ($name:ident, $data:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $expected;
                let mut parser = FormatParser::from(data);
                let parsed = parser.parse_format();
                assert_eq!(parsed, expected);
                assert_eq!(parser.lexer.slice(), "\n");
            }
        };
    }

    parser_test!(
        repeating,
        "(X++(Y..Y))\n",
        Ok(BlockFormat::new(
            LineLayout::RepeatingValue {
                incrementing: "X".into(),
                repeating: "Y".into(),
            },
            None
        ))
    );
    parser_test!(
        repeating_block_kind,
        "(X++(R..R)), XYDATA\n",
        Ok(BlockFormat::new(
            LineLayout::RepeatingValue {
                incrementing: "X".into(),
                repeating: "R".into(),
            },
            Some("XYDATA")
        ))
    );
    parser_test!(
        multi_group,
        "(XY..XY)\n",
        Ok(BlockFormat::new(LineLayout::MultiGroup(vec!["X".into(), "Y".into()]), None))
    );
    parser_test!(
        multi_group_block_kind,
        "(XY..XY), PEAKS\n",
        Ok(BlockFormat::new(
            LineLayout::MultiGroup(vec!["X".into(), "Y".into()]),
            Some("PEAKS")
        ))
    );
    parser_test!(
        single_group,
        "(XYWA)\n",
        Ok(BlockFormat::new(
            LineLayout::SingleGroup(vec!["X".into(), "Y".into(), "W".into(), "A".into()]),
            None
        ))
    );
    parser_test!(
        single_group_block_kind,
        "(XYWA), PEAK ASSIGNMENTS\n",
        Ok(BlockFormat::new(
            LineLayout::SingleGroup(vec!["X".into(), "Y".into(), "W".into(), "A".into()]),
            Some("PEAK ASSIGNMENTS")
        ))
    );
    parser_test!(
        invalid_literal,
        "(X.Y)\n",
        Err(Error::invalid_literal(Position { line: 0, column: 2 }))
    );
    parser_test!(
        empty_repeat,
        "(..X)\n",
        Err(Error::empty_repeat(Position { line: 0, column: 1 }))
    );
    parser_test!(
        mismatched_repeat,
        "(X..)\n",
        Err(Error::mismatched_repeat(Position { line: 0, column: 0 }))
    );
    parser_test!(
        multiple_increment,
        "(X++X++(Y..Y)\n",
        Err(Error::multiple_increment(Position { line: 0, column: 5 }))
    );
    parser_test!(
        multiple_repeat,
        "(X++(Y..Y..Y)\n",
        Err(Error::multiple_repeat(Position { line: 0, column: 9 }))
    );
    parser_test!(
        increment_after_repeat,
        "(Y..Y)X++\n",
        Err(Error::increment_after_repeat(Position { line: 0, column: 7 }))
    );
    parser_test!(
        multiple_identifier_increment,
        "XF1++(Y..Y)\n",
        Err(Error::multiple_identifier_increment(Position { line: 0, column: 3 }))
    );
}
