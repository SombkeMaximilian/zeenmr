use crate::jcampdx::block_format::{FormatParser, LineLayout};
use crate::jcampdx::data::DatasetBuilder;
use crate::jcampdx::decoding::{Decoder, ExitStatus};
use crate::jcampdx::error::{Error, Result};
use crate::jcampdx::{Dataset, Token, Value};
use crate::{Location, Stack};
use logos::{Lexer, Logos};
use std::num::IntErrorKind;

/// Exit status of the key handler.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum KeyExit {
    /// A key was successfully parsed, what follows is its value.
    Success,
    /// The input ended, either the value was empty or a special handler found
    /// the end.
    EndOfInput,
    /// A special handler terminated by encountering the next key token.
    NextKey,
}

/// Delimiters of bounded values.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Delimiter {
    /// Values bounded by parentheses.
    Parentheses,
    /// Values bounded by angle brackets.
    Angle,
}

/// JCAMP-DX file parser.
#[derive(Debug)]
pub(crate) struct Parser<'source> {
    /// Lexer for tokenizing the key-value pairs in JCAMP-DX headers.
    lexer: Lexer<'source, Token>,
    /// Dataset being constructed.
    builder: DatasetBuilder,
    /// Current key.
    current_key: &'source str,
    /// Current value.
    current_value: Value,
    /// Stack for values bounded by delimiters.
    bounded_stack: Stack<Delimiter, Value>,
    /// Concatenate consecutive strings.
    auto_concatenate: bool,
}

impl<'source> From<&'source str> for Parser<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: Token::lexer(value),
            builder: Dataset::builder(),
            current_key: "TITLE",
            current_value: Value::Empty,
            bounded_stack: Stack::new(),
            auto_concatenate: false,
        }
    }
}

impl<'source> From<Lexer<'source, Token>> for Parser<'source> {
    fn from(value: Lexer<'source, Token>) -> Self {
        Self {
            lexer: value,
            builder: Dataset::builder(),
            current_key: "TITLE",
            current_value: Value::Empty,
            bounded_stack: Stack::new(),
            auto_concatenate: false,
        }
    }
}

impl<'source> Parser<'source> {
    pub(crate) fn parse_source(&mut self) -> Result<Dataset> {
        self.initialize()?;

        self.parse_values()
    }

    /// Main loop for parsing values.
    ///
    /// Advances the lexer until an [`End`] token is encountered. The caller
    /// must ensure that the `Parser` is properly initialized to an entry point
    /// ([`Title`] token).
    ///
    /// [`End`]: Token::End
    /// [`Title`]: Token::Title
    fn parse_values(&mut self) -> Result<Dataset> {
        while let Some(token) = self.lexer.next().transpose()? {
            let reset_auto_concatenate = token != Token::Comma;
            match token {
                Token::Key => {
                    let exit_status = loop {
                        match self.key()? {
                            KeyExit::NextKey => continue,
                            other => break other,
                        }
                    };

                    match exit_status {
                        KeyExit::Success => continue,
                        KeyExit::EndOfInput => break,
                        KeyExit::NextKey => unreachable!(),
                    }
                }
                Token::Comma => self.comma(),
                Token::OpenParenthesis => self.start_bounded(Delimiter::Parentheses),
                Token::CloseParenthesis => self.end_bounded(Delimiter::Parentheses)?,
                Token::OpenAngle => self.start_bounded(Delimiter::Angle),
                Token::CloseAngle => self.end_bounded(Delimiter::Angle)?,
                Token::Numeric => self.numeric(),
                // Tokens like Title are only semantically special if they
                // appear immediately after a Key and before an Equals.
                // Otherwise, they are treated as normal string values.
                Token::String
                | Token::Equals
                | Token::Title
                | Token::Tuples
                | Token::Page
                | Token::EncodedBlock
                | Token::GroupedBlock
                | Token::AmbiguousBlock
                | Token::End => self.string(),
            }
            if reset_auto_concatenate {
                self.auto_concatenate = true;
            }
        }

        Ok(std::mem::take(&mut self.builder).finalize())
    }

    /// Takes the current value and replaces it with [`Empty`]
    ///
    /// [`Empty`]: Value::Empty
    fn take_current_value(&mut self) -> Value {
        std::mem::take(&mut self.current_value)
    }

    /// Checks if the entry point is valid.
    ///
    /// An entry point must appear at the start of the input and consists of the
    /// token sequence [`Key`] -> [`Title`] -> [`Equals`].
    ///
    /// # Errors
    ///
    /// This function returns an error if there are any invalid literals or if
    /// there isn't a valid entry point.
    ///
    /// [`Key`]: Token::Key
    /// [`Title`]: Token::Title
    /// [`Equals`]: Token::Equals
    fn initialize(&mut self) -> Result<()> {
        match (
            self.lexer.next().transpose()?,
            self.lexer.next().transpose()?,
            self.lexer.next().transpose()?,
        ) {
            (Some(Token::Key), Some(Token::Title), Some(Token::Equals)) => Ok(()),
            _ => Err(Error::no_entry_point(self.lexer.location())),
        }
    }

    /// Key handler.
    ///
    /// Advances the lexer until it finds an [`Equals`] token. If there is only
    /// a single special token ([`Title`], etc.) between the initial [`Key`]
    /// token and the [`Equals`] token, calls the respective handlers.
    /// Otherwise, the slice between them becomes the current key.
    ///
    /// [`Key`]: Token::Key
    /// [`Equals`]: Token::Equals
    ///
    /// # Errors
    ///
    /// Returns an error if it finds another [`Key`] token, if the key string is
    /// empty, or if the input ends before finding an [`Equals`] tokens.
    ///
    /// [`Key`]: Token::Key
    /// [`Equals`]: Token::Equals
    fn key(&mut self) -> Result<KeyExit> {
        let current_value = self.take_current_value();
        self.builder
            .insert_parameter(self.current_key, current_value);
        let start = self.lexer.span().end;
        let mut token_count = 0;
        let mut found_equals = false;
        let mut special = None;
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                Token::Key => return Err(Error::multiple_key_tokens(self.lexer.location())),
                Token::Equals => {
                    found_equals = true;
                    break;
                }
                Token::Title
                | Token::Tuples
                | Token::Page
                | Token::EncodedBlock
                | Token::GroupedBlock
                | Token::AmbiguousBlock
                | Token::End => special = Some(token),
                _ => {}
            }
            token_count += 1;
        }
        if !found_equals {
            return Err(Error::end_of_input(self.lexer.location()));
        }
        if token_count == 0 {
            return Err(Error::empty_key(self.lexer.location()));
        }
        if token_count > 1 {
            special = None;
        }
        match special {
            Some(Token::Key) | Some(Token::Equals) => unreachable!(),
            Some(Token::Title) => self.title()?,
            Some(Token::Tuples) => self.tuples(),
            Some(Token::Page) => self.page(),
            Some(Token::EncodedBlock) => {
                return match self.encoded_block()? {
                    ExitStatus::EndOfInput => Ok(KeyExit::EndOfInput),
                    ExitStatus::HeaderKey => Ok(KeyExit::NextKey),
                };
            }
            Some(Token::GroupedBlock) => self.grouped_block(),
            Some(Token::AmbiguousBlock) => self.ambiguous_block(),
            Some(Token::End) => return Ok(KeyExit::EndOfInput),
            Some(Token::Comma)
            | Some(Token::OpenParenthesis)
            | Some(Token::CloseParenthesis)
            | Some(Token::OpenAngle)
            | Some(Token::CloseAngle)
            | Some(Token::Numeric)
            | Some(Token::String)
            | None => {
                let end = self.lexer.span().start;
                self.current_key = self.lexer.source()[start..end].trim();
            }
        }

        Ok(KeyExit::Success)
    }

    /// Handles comma separators.
    ///
    /// A comma disables automatic concatenation. If concatenation is already
    /// disabled, i.e., consecutive commas are encountered, inserts an empty
    /// value into the current context.
    ///
    /// [`Empty`]: Value::Empty
    /// [`Comma`]: Token::Comma
    fn comma(&mut self) {
        if self.auto_concatenate {
            self.auto_concatenate = false;
        } else {
            if let Some(top) = self.bounded_stack.top_mut() {
                top.values.push(Value::Empty);
            } else {
                match self.current_value {
                    Value::Empty => self.current_value = Value::Array(vec![Value::Empty]),
                    Value::Array(ref mut array) => array.push(Value::Empty),
                    _ => {
                        self.current_value =
                            Value::Array(vec![self.take_current_value(), Value::Empty])
                    }
                }
            }
        }
    }

    /// Adds a [`Frame`] to the [`Stack`] with the encountered delimiter kind.
    ///
    /// [`Frame`]: crate::Frame
    fn start_bounded(&mut self, delimiter: Delimiter) {
        self.bounded_stack
            .push(delimiter, self.lexer.location());
    }

    /// Finalizes the [`Frame`] at the top of the [`Stack`].
    ///
    /// [`Frame`]: crate::Frame
    ///
    /// If the stack is not empty, appends the value to the enclosing [`Frame`],
    /// allowing nested bounded structures. Otherwise, stores the resulting
    /// [`Value`] in the current value being built by the `Parser`.
    ///
    /// [`Frame`]: crate::Frame
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Delimiter`]s are mismatched, i.e., the
    /// encountered delimiter does not match the one at the top of the
    /// [`Stack`].
    fn end_bounded(&mut self, delimiter: Delimiter) -> Result<()> {
        if self.bounded_stack.top_delimiter() != Some(&delimiter) {
            return Err(Error::mismatched_delimiter(self.lexer.location()));
        }
        let frame = self
            .bounded_stack
            .pop()
            .ok_or_else(|| Error::mismatched_delimiter(self.lexer.location()))?;
        let value = match frame.values.len() {
            0 => Value::Empty,
            1 => frame.values.into_iter().next().unwrap(),
            _ => Value::Array(frame.values),
        };
        if let Some(top) = self.bounded_stack.top_mut() {
            top.values.push(value);
        } else {
            match self.current_value {
                Value::Empty => self.current_value = Value::Array(vec![value]),
                Value::Array(ref mut array) => array.push(value),
                _ => self.current_value = Value::Array(vec![self.take_current_value(), value]),
            }
        }

        Ok(())
    }

    /// Parses [`Numeric`] tokens.
    ///
    /// [`Numeric`]: Token::Numeric
    ///
    /// Attempts to parse the token as `i64`. If parsing fails due to overflow,
    /// records a non-fatal [`Overflow`] error and uses [`i64::MIN`] as the
    /// value. Otherwise, falls back to parsing as `f64`.
    ///
    /// [`Overflow`]: crate::jcampdx::error::Kind::Overflow;
    fn numeric(&mut self) {
        let value = match self.lexer.slice().parse::<i64>() {
            Ok(int) => Value::Integer(int),
            Err(e) => match e.kind() {
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                    self.builder
                        .push_error(Error::overflow(self.lexer.location(), e));

                    Value::Integer(i64::MIN)
                }
                _ => Value::Float(
                    self.lexer
                        .slice()
                        .parse::<f64>()
                        .expect("lexer (regex) matched non-numeric"),
                ),
            },
        };
        if let Some(top) = self.bounded_stack.top_mut() {
            top.values.push(value);
        } else {
            match self.current_value {
                Value::Empty => self.current_value = value,
                Value::Array(ref mut array) => array.push(value),
                _ => self.current_value = Value::Array(vec![self.take_current_value(), value]),
            }
        }
    }

    /// Handles [`String`] tokens.
    ///
    /// Inserts the string into the current context. If automatic concatenation
    /// is enabled and the previous value is also a string, the values are
    /// concatenated with a separating space. Otherwise, a new string value is
    /// inserted.
    ///
    /// [`String`]: Token::String
    fn string(&mut self) {
        let value = self.lexer.slice();
        let mut push_string = |values: &mut Vec<Value>| {
            if let Some(Value::String(previous)) = values.last_mut()
                && self.auto_concatenate
            {
                if !previous.is_empty() {
                    previous.push(' ');
                }
                previous.push_str(value);
            } else {
                values.push(Value::String(value.to_string()));
            }
        };
        if let Some(top) = self.bounded_stack.top_mut() {
            push_string(&mut top.values);
        } else {
            match self.current_value {
                Value::Empty => self.current_value = Value::String(value.to_string()),
                Value::String(ref mut previous) if self.auto_concatenate => {
                    if previous.len() > 0 {
                        previous.push(' ');
                    }
                    previous.push_str(value);
                }
                Value::Array(ref mut array) => push_string(array),
                _ => {
                    self.current_value = Value::Array(vec![
                        self.take_current_value(),
                        Value::String(value.to_string()),
                    ]);
                }
            }
        }
    }

    /// Handles a [`Title`] token by starting a `Parser` child.
    ///
    /// [`Title`]: Token::Title
    ///
    /// JCAMP-DX files can recursively contain child datasets, which are
    /// [`Title`] and [`End`] token pairs within another [`Title`] and [`End`]
    /// token pair.
    ///
    /// [`Title`]: Token::Title
    /// [`End`]: Token::End
    ///
    /// # Errors
    ///
    /// Returns an error if the child `Parser` encounters a fatal error.
    fn title(&mut self) -> Result<()> {
        let mut sub_parser = Self::from(self.lexer.clone());
        let child_dataset = sub_parser.parse_values()?;
        self.builder.push_child(child_dataset);
        self.lexer = sub_parser.lexer;

        Ok(())
    }

    fn tuples(&mut self) {}

    fn page(&mut self) {}

    /// Handles [`EncodedBlock`] tokens.
    ///
    /// [`EncodedBlock`]: Token::EncodedBlock
    ///
    /// Blocks of encoded data require context switches, once to extract the
    /// identifiers of the data and once for the decoding itself.
    ///
    /// # Identifiers
    ///
    /// Encoded blocks use the `XYDATA` format, which is followed by a string
    /// of the form `X++(Y..Y)`, where `X` is the positional or independent
    /// variable, while `Y` is the intensity or dependent variable. Common
    /// examples include `X` and `F1` for the independent variable, and `Y`, `R`
    /// and `I` for the dependent variable. See [`FormatParser`] for more
    /// information.
    ///
    /// # Encoding
    ///
    /// See the [official standard](http://jcamp-dx.org/protocols/dxir01.pdf)
    /// for information about the JCAMP-DX `ASDF` encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Decoder`] encounters a fatal error.
    fn encoded_block(&mut self) -> Result<ExitStatus> {
        let mut format_parser = FormatParser::from(self.lexer.clone());
        let identifiers = match format_parser.parse_format() {
            Ok(format) => match format.line_layout {
                LineLayout::RepeatingValue {
                    incrementing,
                    repeating,
                } => Some((incrementing, repeating)),
                _ => {
                    self.builder
                        .push_error(Error::mismatched_block_format(self.lexer.location()));

                    None
                }
            },
            Err(e) => {
                self.builder.push_error(e.into());

                None
            }
        };
        let mut decoder = Decoder::from(format_parser.into_lexer());
        if let Some((incrementing, repeating)) = identifiers {
            decoder.set_incrementing(incrementing);
            decoder.set_repeating(repeating);
        }
        let decoded_block = decoder.decode_source()?;
        self.lexer = decoder.into_lexer().morph();
        self.builder
            .extend_errors(decoded_block.errors.into_iter().map(|e| e.into()));
        self.builder.push_table(decoded_block.table);

        Ok(decoded_block.exit)
    }

    fn grouped_block(&mut self) {}

    fn ambiguous_block(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::read_to_string;
    use std::path::PathBuf;

    fn workspace_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    macro_rules! parser_test {
        ($name:ident, $version:tt, $file:tt) => {
            #[test]
            fn $name() {
                let path = workspace_dir()
                    .join("data")
                    .join("jcamp-dx")
                    .join("test")
                    .join($version)
                    .join($file);
                let content = read_to_string(path).unwrap();
                let parsed = Parser::from(content.as_str())
                    .parse_source()
                    .unwrap();
            }
        };
    }

    parser_test!(parser_v5_ntuples_affn, "v5", "ntuples_affn.dx");
    parser_test!(parser_v5_ntuples_pac, "v5", "ntuples_pac.dx");
    parser_test!(parser_v5_ntuples_sqz, "v5", "ntuples_sqz.dx");
    parser_test!(parser_v5_ntuples_difdup, "v5", "ntuples_difdup.dx");
    parser_test!(parser_v5_xydata_affn, "v5", "xydata_affn.dx");
    parser_test!(parser_v5_xydata_pac, "v5", "xydata_pac.dx");
    parser_test!(parser_v5_xydata_sqz, "v5", "xydata_sqz.dx");
    parser_test!(parser_v5_xydata_difdup, "v5", "xydata_difdup.dx");
    parser_test!(parser_v6_ntuples_affn, "v6", "ntuples_affn.dx");
    parser_test!(parser_v6_ntuples_pac, "v6", "ntuples_pac.dx");
    parser_test!(parser_v6_ntuples_sqz, "v6", "ntuples_sqz.dx");
    parser_test!(parser_v6_ntuples_difdup, "v6", "ntuples_difdup.dx");
    parser_test!(parser_v6_xydata_affn, "v6", "xydata_affn.dx");
    parser_test!(parser_v6_xydata_pac, "v6", "xydata_pac.dx");
    parser_test!(parser_v6_xydata_sqz, "v6", "xydata_sqz.dx");
    parser_test!(parser_v6_xydata_difdup, "v6", "xydata_difdup.dx");
}
