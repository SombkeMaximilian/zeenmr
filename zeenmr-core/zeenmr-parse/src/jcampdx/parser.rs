use crate::jcampdx::error::{Error, Result};
use crate::jcampdx::{Token, Value};
use crate::{Location, Stack};
use logos::{Lexer, Logos};
use std::collections::HashMap;

/// Delimiters of bounded values.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Delimiter {
    /// Values bounded by parentheses are arrays.
    Parentheses,
    /// Values bounded by angle brackets are strings.
    Angle(usize),
}

/// JCAMP-DX file parser.
#[derive(Debug)]
pub(crate) struct Parser<'source> {
    /// Lexer for tokenizing the key-value pairs in JCAMP-DX headers.
    lexer: Lexer<'source, Token>,
    /// Parameters being constructed.
    parameters: HashMap<String, Value>,
    /// Current key.
    current_key: &'source str,
    /// Current value.
    current_value: Value,
    /// Stack for values bounded by delimiters.
    bounded_stack: Stack<Delimiter, Value>,
    /// Concatenate consecutive strings.
    auto_concatenate: bool,
    /// Non-fatal errors during parsing.
    errors: Vec<Error>,
}

impl<'source> From<&'source str> for Parser<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: Token::lexer(value),
            parameters: HashMap::new(),
            current_key: "TITLE",
            current_value: Value::Empty,
            bounded_stack: Stack::new(),
            auto_concatenate: false,
            errors: Vec::new(),
        }
    }
}

impl<'source> Parser<'source> {
    pub(crate) fn parse_source(mut self) -> Result<()> {
        self.initialize()?;

        while let Some(token) = self.lexer.next() {
            let clear_auto_concatenate = token != Ok(Token::Comma);
            match token {
                Ok(Token::Key) => self.key()?,
                Ok(Token::Comma) => self.comma(),
                Ok(Token::OpenParenthesis) => self.open_parenthesis(),
                Ok(Token::CloseParenthesis) => self.close_parenthesis(),
                Ok(Token::OpenAngle) => self.open_angle(),
                Ok(Token::CloseAngle) => self.close_angle(),
                Ok(Token::Numeric) => self.numeric(),
                // Tokens like Title are only semantically special if they
                // appear immediately after a Key and before an Equals.
                // Otherwise, they are treated as normal string values.
                Ok(Token::String)
                | Ok(Token::Equals)
                | Ok(Token::Title)
                | Ok(Token::Tuples)
                | Ok(Token::Page)
                | Ok(Token::EncodedBlock)
                | Ok(Token::GroupedBlock)
                | Ok(Token::AmbiguousBlock)
                | Ok(Token::End) => self.string(),
                Err(e) => panic!("{e}"),
            }
            if clear_auto_concatenate {
                self.auto_concatenate = false;
            }
        }

        Ok(())
    }

    /// Returns `true` if not inside a bounded value.
    fn is_bounded(&self) -> bool {
        !self.bounded_stack.is_empty()
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
    fn key(&mut self) -> Result<()> {
        let current_value = self.take_current_value();
        self.parameters.insert(self.current_key.to_string(), current_value);
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
            Some(Token::Title) => self.title(),
            Some(Token::Tuples) => self.tuples(),
            Some(Token::Page) => self.page(),
            Some(Token::EncodedBlock) => self.encoded_block(),
            Some(Token::GroupedBlock) => self.grouped_block(),
            Some(Token::AmbiguousBlock) => self.ambiguous_block(),
            Some(Token::End) => self.end(),
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

        Ok(())
    }

    fn comma(&mut self) {}

    fn open_parenthesis(&mut self) {}

    fn close_parenthesis(&mut self) {}

    fn open_angle(&mut self) {}

    fn close_angle(&mut self) {}

    fn numeric(&mut self) {}

    fn string(&mut self) {}

    fn title(&mut self) {}

    fn tuples(&mut self) {}

    fn page(&mut self) {}

    fn encoded_block(&mut self) {}

    fn grouped_block(&mut self) {}

    fn ambiguous_block(&mut self) {}

    fn end(&mut self) {}
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
                let parsed = Parser::from(content.as_str()).parse_source();
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
