use crate::jcampdx::block_format::error::{Error, Result};
use crate::jcampdx::block_format::{BlockFormat, FormatToken, LineLayout};
use crate::{Cursor, Location, Position};
use logos::{Lexer, Logos};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum State {
    Prefix,
    AfterIncrement,
    Suffix,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Termination {
    EndToken,
    EndOfInput,
}

#[derive(Debug)]
pub(crate) struct FormatParser<'source> {
    lexer: Lexer<'source, FormatToken>,
    state: State,
    termination: Termination,
    start: Position,
    prefix: Vec<&'source str>,
    suffix: Vec<&'source str>,
    increment: Vec<&'source str>,
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
            termination: Termination::EndOfInput,
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
            termination: Termination::EndOfInput,
        }
    }
}

impl<'source> FormatParser<'source> {
    pub(crate) fn parse_format(&mut self) -> Result<BlockFormat> {
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                FormatToken::Identifier => self.identifier()?,
                FormatToken::Increment => self.increment()?,
                FormatToken::Repeat => self.repeat()?,
                FormatToken::DataBlockKind => {
                    self.data_block_kind()?;
                    self.termination = Termination::EndToken;
                    break;
                }
                FormatToken::End => {
                    self.termination = Termination::EndToken;
                    break;
                }
            }
        }

        match self.termination {
            Termination::EndToken => self.finalize(),
            Termination::EndOfInput => Err(Error::end_of_input(self.lexer.location())),
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
                        increment,
                        repeating,
                    },
                    self.block_kind,
                ))
            }
            State::Suffix => {
                if self.prefix != self.suffix {
                    return Err(Error::mismatched_repeat(self.lexer.location()));
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
                            increment,
                            repeating,
                        },
                        self.block_kind,
                    ))
                }
            }
        }
    }

    fn identifier(&mut self) -> Result<()> {
        match self.state {
            State::Prefix | State::AfterIncrement => {
                self.prefix.push(self.lexer.slice());

                Ok(())
            }
            State::Suffix => {
                self.suffix.push(self.lexer.slice());

                Ok(())
            }
        }
    }

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
            State::AfterIncrement | State::Suffix => {
                Err(Error::increment_after_repeat(self.lexer.location()))
            }
        }
    }

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

    fn data_block_kind(&mut self) -> Result<()> {
        match self.block_kind {
            Some(_) => unreachable!(),
            None => {
                self.block_kind = Some(&self.lexer.slice()[1..].trim());

                if let Some(next) = self.lexer.next().transpose()? {
                    match next {
                        FormatToken::End => Ok(()),
                        _ => Err(Error::token_after_block_kind(self.lexer.location())),
                    }
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
                let parsed = parser.parse_format().unwrap();
                assert_eq!(parsed, expected);
            }
        };
    }

    parser_test!(
        repeating,
        "(X++(Y..Y))\n",
        BlockFormat::new(
            LineLayout::RepeatingValue {
                increment: "X".into(),
                repeating: "Y".into(),
            },
            None
        )
    );
    parser_test!(
        repeating_block_kind,
        "(X++(R..R)), XYDATA\n",
        BlockFormat::new(
            LineLayout::RepeatingValue {
                increment: "X".into(),
                repeating: "R".into(),
            },
            Some("XYDATA")
        )
    );
    parser_test!(
        multi_group,
        "(XY..XY)\n",
        BlockFormat::new(LineLayout::MultiGroup(vec!["X".into(), "Y".into()]), None)
    );
    parser_test!(
        multi_group_block_kind,
        "(XY..XY), PEAKS\n",
        BlockFormat::new(
            LineLayout::MultiGroup(vec!["X".into(), "Y".into()]),
            Some("PEAKS")
        )
    );
    parser_test!(
        single_group,
        "(XYWA)\n",
        BlockFormat::new(
            LineLayout::SingleGroup(vec!["X".into(), "Y".into(), "W".into(), "A".into()]),
            None
        )
    );
    parser_test!(
        single_group_block_kind,
        "(XYWA), PEAK ASSIGNMENTS\n",
        BlockFormat::new(
            LineLayout::SingleGroup(vec!["X".into(), "Y".into(), "W".into(), "A".into()]),
            Some("PEAK ASSIGNMENTS")
        )
    );
}
