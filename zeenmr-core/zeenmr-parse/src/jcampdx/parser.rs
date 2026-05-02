use crate::Stack;
use crate::data::{Dataset, ParameterTable, Value};
use crate::jcampdx::JcampDxDataset;
use crate::jcampdx::block_format::{FormatParser, LineLayout};
use crate::jcampdx::decoding::Decoder;
use crate::jcampdx::error::{Error, Result};
use crate::jcampdx::lexer::Token;
use crate::jcampdx::tabulation::TableParser;
use logos::{Lexer, Logos};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::IntErrorKind;

/// Exit status of a child parser.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) enum ChildParserExit {
    /// Child parser was terminated by the end of the input.
    #[default]
    EndOfInput,
    /// Child parser was terminated by encountering its respective end token.
    EndToken,
}

/// Delimiters of bounded values.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Delimiter {
    /// Values bounded by parentheses.
    Parentheses,
    /// Values bounded by angle brackets.
    Angle,
}

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

/// Marker for normal parser mode.
#[derive(Debug)]
pub(crate) struct Normal;

/// Marker for `NTUPLES` parser mode.
#[derive(Debug)]
pub(crate) struct Tuples;

/// JCAMP-DX file parser.
#[derive(Debug)]
pub(crate) struct Parser<'source, M> {
    /// Lexer for tokenizing the key-value pairs in JCAMP-DX headers.
    lexer: Lexer<'source, Token>,
    /// Dataset being constructed.
    dataset: JcampDxDataset<'source>,
    /// Current key.
    current_key: Option<&'source str>,
    /// Current value.
    current_value: Value<'source>,
    /// Stack for values bounded by delimiters.
    bounded_stack: Stack<Delimiter, Value<'source>>,
    /// Concatenate consecutive strings.
    auto_concatenate: bool,
    /// Parsing mode.
    mode: PhantomData<M>,
}

impl<'source> From<&'source str> for Parser<'source, Normal> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: Token::lexer(value),
            dataset: Dataset::default(),
            current_key: None,
            current_value: Value::Empty,
            bounded_stack: Stack::new(),
            auto_concatenate: false,
            mode: PhantomData,
        }
    }
}

impl<'source, M> From<Lexer<'source, Token>> for Parser<'source, M> {
    fn from(value: Lexer<'source, Token>) -> Self {
        Self {
            lexer: value,
            dataset: Dataset::default(),
            current_key: None,
            current_value: Value::Empty,
            bounded_stack: Stack::new(),
            auto_concatenate: false,
            mode: PhantomData,
        }
    }
}

/// Trait for methods that diverge between parser modes.
pub(crate) trait ParserMode {
    /// Inserts a parameter into the appropriate table.
    fn insert_parameter(&mut self);

    /// Handles [`Title`] tokens.
    ///
    /// [`Title`]: Token::Title
    fn title(&mut self) -> Result<()>;

    /// Handles [`Tuples`] tokens.
    ///
    /// [`Tuples`]: Token::Tuples
    fn tuples(&mut self) -> Result<()>;

    /// Handles [`Page`] tokens.
    ///
    /// [`Page`]: Token::Page
    fn page(&mut self) -> Result<()>;
}

impl<'source> ParserMode for Parser<'source, Normal> {
    /// Inserts the current key-value pair into the dataset's parameter table if
    /// `current_key` is a `Some` value.
    ///
    /// This will never insert a key-value pair into the data parameters.
    fn insert_parameter(&mut self) {
        if let Some(key) = self.current_key.take() {
            let value = self.take_current_value();
            self.dataset.parameters.insert(key.into(), value);
        }
    }

    /// Handles [`Title`] tokens by starting a `Normal` child parser.
    ///
    /// [`Title`]: Token::Title
    ///
    /// JCAMP-DX files can recursively contain child datasets, which are
    /// [`Title`] and [`End`] token pairs within another [`Title`] and [`End`]
    /// token pair. This does not return an exit code because any main loop
    /// would simply terminate regardless.
    ///
    /// [`Title`]: Token::Title
    /// [`End`]: Token::End
    ///
    /// # Errors
    ///
    /// Returns an error if the child parser encounters a fatal error.
    fn title(&mut self) -> Result<()> {
        let mut sub_parser = Parser::<Normal>::from(self.lexer.clone());
        sub_parser.current_key = Some("TITLE");
        let child_dataset = sub_parser.parse_values()?;
        self.dataset.children.push(child_dataset);
        self.lexer = sub_parser.lexer;

        Ok(())
    }

    /// Handles [`Tuples`] tokens by starting a `Tuples` child parser.
    ///
    /// [`Tuples`]: Token::Tuples
    ///
    /// # Errors
    ///
    /// Returns an error if the child parser encounters a fatal error.
    fn tuples(&mut self) -> Result<()> {
        let mut sub_parser = Parser::<Tuples>::from(self.lexer.clone());
        sub_parser.current_key = Some("NTUPLES");
        let child_dataset = sub_parser.parse_values()?;
        self.dataset.children.push(child_dataset);
        self.lexer = sub_parser.lexer;

        Ok(())
    }

    /// Handles [`Page`] tokens.
    ///
    /// [`Page`]: Token::Page
    ///
    /// [`Page`] tokens are not valid outside an `NTUPLES` block. Therefore,
    /// reaching this point is always an error that cannot be recovered from,
    /// as the file is almost guaranteed to be malformed or corrupted.
    fn page(&mut self) -> Result<()> {
        Err(Error::unexpected_page(self.lexer.span().into()))
    }
}

impl<'source> ParserMode for Parser<'source, Tuples> {
    /// Inserts the current key-value pair into the active parameter table if
    /// `current_key` is a `Some` value.
    ///
    /// If at least one [`Page`] token has been encountered in the current
    /// context (i.e., until the current parser terminates), this will be the
    /// most recently added data-specific parameter table.
    fn insert_parameter(&mut self) {
        if let Some(key) = self.current_key.take() {
            let value = self.take_current_value();
            if let Some(parameters) = self.dataset.data_parameters.last_mut() {
                parameters.insert(key.into(), value);
            } else {
                self.dataset.parameters.insert(key.into(), value);
            }
        }
    }

    /// Handles [`Title`] tokens.
    ///
    /// [`Title`]: Token::Title
    ///
    /// Further nesting is not valid inside an `NTUPLES` block. Therefore,
    /// reaching this point is always an error that cannot be recovered from,
    /// as the file is almost guaranteed to be malformed or corrupted.
    fn title(&mut self) -> Result<()> {
        Err(Error::nested_tuples(self.lexer.span().into()))
    }

    /// Handles [`Tuples`] tokens.
    ///
    /// [`Tuples`]: Token::Tuples
    ///
    /// Further nesting is not valid inside an `NTUPLES` block. Therefore,
    /// reaching this point is always an error that cannot be recovered from,
    /// as the file is almost guaranteed to be malformed or corrupted.
    fn tuples(&mut self) -> Result<()> {
        Err(Error::nested_tuples(self.lexer.span().into()))
    }

    /// Handles [`Page`] tokens.
    ///
    /// [`Page`]: Token::Page
    ///
    /// Adds a new, empty data-specific parameter table to the dataset and
    /// switches to the key handler for the inline page assignment (e.g., `N=1`,
    /// or `F1=4370.000`).
    ///
    /// # Errors
    ///
    /// Returns an error if the following key handler fails, or if the key
    /// handler returns an end of input exit code.
    fn page(&mut self) -> Result<()> {
        self.dataset
            .data_parameters
            .push(ParameterTable::default());
        let exit_status = loop {
            match self.key()? {
                KeyExit::NextKey => continue,
                other => break other,
            }
        };

        match exit_status {
            KeyExit::Success => Ok(()),
            KeyExit::EndOfInput => Err(Error::end_of_input(self.lexer.span().into())),
            KeyExit::NextKey => unreachable!(),
        }
    }
}

impl<'source, M> Parser<'source, M>
where
    Parser<'source, M>: ParserMode,
{
    /// Parses the source until the last matching [`End`] token or the end of
    /// the source.
    ///
    /// [`End`]: Token::End
    pub(crate) fn parse_source(&mut self) -> Result<JcampDxDataset<'source>> {
        self.initialize()?;
        self.current_key = Some("TITLE");

        self.parse_values()
    }

    /// Checks if the entry point is valid.
    ///
    /// An entry point must appear at the start of the input and consists of the
    /// token sequence [`Key`] -> [`Title`] -> [`Equals`].
    ///
    /// [`Key`]: Token::Key
    /// [`Title`]: Token::Title
    /// [`Equals`]: Token::Equals
    ///
    /// # Errors
    ///
    /// This function returns an error if there are any invalid literals or if
    /// there isn't a valid entry point.
    fn initialize(&mut self) -> Result<()> {
        match (
            self.lexer.next().transpose()?,
            self.lexer.next().transpose()?,
            self.lexer.next().transpose()?,
        ) {
            (Some(Token::Key), Some(Token::Title), Some(Token::Equals)) => Ok(()),
            _ => Err(Error::no_entry_point(self.lexer.span().into())),
        }
    }

    /// Main loop for parsing values.
    ///
    /// Advances the [`Lexer`] until an [`End`] token is encountered. The caller
    /// must ensure that the `Parser` is properly initialized to an entry point
    /// ([`Title`] token).
    ///
    /// [`End`]: Token::End
    /// [`Title`]: Token::Title
    fn parse_values(&mut self) -> Result<JcampDxDataset<'source>> {
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
        self.insert_parameter();

        Ok(std::mem::take(&mut self.dataset))
    }

    /// Takes the current value and replaces it with [`Empty`]
    ///
    /// [`Empty`]: Value::Empty
    fn take_current_value(&mut self) -> Value<'source> {
        std::mem::take(&mut self.current_value)
    }

    /// Handles [`Key`] tokens.
    ///
    /// [`Key`]: Token::Key
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
        while let Some(top) = self.bounded_stack.top() {
            self.dataset
                .errors
                .push(Error::unclosed_delimiter(top.start));
            self.end_bounded(top.delimiter)
                .expect("should always get the right delimiter");
        }
        self.insert_parameter();
        let start = self.lexer.span().end;
        let mut token_count = 0;
        let mut found_equals = false;
        let mut special = None;
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                Token::Key => return Err(Error::multiple_key_tokens(self.lexer.span().into())),
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
            return Err(Error::end_of_input(self.lexer.span().into()));
        }
        if token_count == 0 {
            return Err(Error::empty_key(self.lexer.span().into()));
        }
        if token_count > 1 {
            special = None;
        }
        match special {
            Some(Token::Key) | Some(Token::Equals) => unreachable!(),
            Some(Token::Title) => self.title()?,
            Some(Token::Tuples) => self.tuples()?,
            Some(Token::Page) => self.page()?,
            Some(Token::EncodedBlock) => {
                return match self.encoded_block()? {
                    ChildParserExit::EndOfInput => Ok(KeyExit::EndOfInput),
                    ChildParserExit::EndToken => Ok(KeyExit::NextKey),
                };
            }
            Some(Token::GroupedBlock) => {
                return match self.grouped_block()? {
                    ChildParserExit::EndOfInput => Ok(KeyExit::EndOfInput),
                    ChildParserExit::EndToken => Ok(KeyExit::NextKey),
                };
            }
            Some(Token::AmbiguousBlock) => {
                return match self.ambiguous_block()? {
                    ChildParserExit::EndOfInput => Ok(KeyExit::EndOfInput),
                    ChildParserExit::EndToken => Ok(KeyExit::NextKey),
                };
            }
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
                self.current_key = Some(self.lexer.source()[start..end].trim());
            }
        }

        Ok(KeyExit::Success)
    }

    /// Handles [`Comma`] tokens.
    ///
    /// [`Comma`]: Token::Comma
    ///
    /// A comma disables automatic concatenation. If concatenation is already
    /// disabled, i.e., consecutive [`Comma`]s are encountered, inserts an
    /// [`Empty`] value into the current context.
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
            .push(delimiter, self.lexer.span().into());
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
            return Err(Error::mismatched_delimiter(self.lexer.span().into()));
        }
        let frame = self
            .bounded_stack
            .pop()
            .expect("should not be empty due to successful top delimiter check");
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
                    self.dataset
                        .errors
                        .push(Error::overflow(self.lexer.span().into()));

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
        let push_string = |values: &mut Vec<Value<'source>>| {
            if let Some(Value::String(previous)) = values.last_mut()
                && self.auto_concatenate
            {
                let mut owned = previous.clone().into_owned();
                if !owned.is_empty() {
                    owned.push(' ');
                }
                owned.push_str(value);
                *previous = Cow::Owned(owned);
            } else {
                values.push(Value::from(value));
            }
        };
        if let Some(top) = self.bounded_stack.top_mut() {
            push_string(&mut top.values);
        } else {
            match self.current_value {
                Value::Empty => self.current_value = Value::from(value),
                Value::String(ref mut previous) if self.auto_concatenate => {
                    let mut owned = previous.clone().into_owned();
                    if !owned.is_empty() {
                        owned.push(' ');
                    }
                    owned.push_str(value);
                    *previous = Cow::Owned(owned);
                }
                Value::Array(ref mut array) => push_string(array),
                _ => {
                    self.current_value =
                        Value::Array(vec![self.take_current_value(), Value::from(value)]);
                }
            }
        }
    }

    /// Handles [`EncodedBlock`] tokens.
    ///
    /// [`EncodedBlock`]: Token::EncodedBlock
    ///
    /// Blocks of encoded data require context switches, once to extract the
    /// identifiers of the data and once for the decoding itself.
    ///
    /// # Identifiers
    ///
    /// Encoded blocks use the `XYDATA` format, which is of the form
    /// `X++(Y..Y)`, where `X` is the positional or independent variable, while
    /// `Y` is the intensity or dependent variable. Common examples include `X`
    /// and `F1` for the independent variable, and `Y`, `R` and `I` for the
    /// dependent variable. See [`FormatParser`] for more information.
    ///
    /// # Encoding
    ///
    /// See the [official standard](http://jcamp-dx.org/protocols/dxir01.pdf)
    /// for information about the JCAMP-DX `ASDF` encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Decoder`] encounters a fatal error.
    fn encoded_block(&mut self) -> Result<ChildParserExit> {
        let mut format_parser = FormatParser::from(self.lexer.clone());
        let format = format_parser.parse_format();
        let mut decoder = Decoder::from(format_parser.into_lexer());
        match format {
            Ok(format) => {
                if let Some(kind) = format.kind {
                    decoder.set_title(kind)
                }
                match format.line_layout {
                    LineLayout::RepeatingValue {
                        incrementing,
                        repeating,
                    } => {
                        decoder.set_incrementing(incrementing);
                        decoder.set_repeating(repeating);
                    }
                    _ => {
                        self.dataset
                            .errors
                            .push(Error::mismatched_block_format(self.lexer.span().into()));
                    }
                }
            }
            Err(e) => {
                self.dataset.errors.push(e.into());
            }
        }
        let decoded_block = decoder.decode_source()?;
        self.lexer = decoder.into_lexer().morph();
        self.dataset
            .errors
            .extend(decoded_block.errors.into_iter().map(Error::from));
        self.dataset.data_tables.push(decoded_block.table);

        Ok(decoded_block.exit)
    }

    /// Handles [`GroupedBlock`] tokens.
    ///
    /// [`GroupedBlock`]: Token::GroupedBlock
    ///
    /// Blocks of grouped data require context switches, once to extract the
    /// identifiers of the data and once for transposing the groups to columns.
    ///
    /// # Identifiers
    ///
    /// Grouped blocks use any formats other than `XYDATA`, which are of the
    /// form `(G)` or `(G..G)`, where `G` may be any group of identifiers.
    /// Common examples include `X`, `Y`, `R` and `I` for measurement data, as
    /// well as `W` for peak width, `M` for multiplicity.
    ///
    /// # Errors
    ///
    /// Returns an error if the format of the block couldn't be parsed, or if
    /// the [`TableParser`] encounters a fatal error.
    fn grouped_block(&mut self) -> Result<ChildParserExit> {
        let mut format_parser = FormatParser::from(self.lexer.clone());
        let format = format_parser.parse_format()?;
        let mut table_parser = match format.line_layout {
            LineLayout::RepeatingValue { .. } => {
                return Err(Error::mismatched_block_format(self.lexer.span().into()));
            }
            LineLayout::GroupedValues(identifiers) => {
                TableParser::from(format_parser.into_lexer()).with_identifiers(identifiers)
            }
        };
        if let Some(kind) = format.kind {
            table_parser.set_title(kind);
        }
        let tabulated_block = table_parser.tabulate_source()?;
        self.lexer = table_parser.into_lexer().morph();
        self.dataset.errors.extend(
            tabulated_block
                .errors
                .into_iter()
                .map(Error::from),
        );
        self.dataset
            .data_tables
            .push(tabulated_block.table);

        Ok(tabulated_block.exit)
    }

    /// Handles [`AmbiguousBlock`] tokens.
    ///
    /// [`AmbiguousBlock`]: Token::AmbiguousBlock
    ///
    /// Blocks of this kind may be either encoded or grouped, which can be
    /// determined by the format specifier. Either kind requires context
    /// switches, once to extract the format specifier itself, and once for
    /// parsing the data.
    ///
    /// # Format Specifier
    ///
    /// Encoded blocks use the `XYDATA` format, which is of the form
    /// `X++(Y..Y)`, where `X` is the positional or independent variable, while
    /// `Y` is the intensity or dependent variable. Common examples include `X`
    /// and `F1` for the independent variable, and `Y`, `R` and `I` for the
    /// dependent variable.
    ///
    /// Grouped blocks use the other formats, which are of the form `(G)` or
    /// `(G..G)`, where `G` may be any group of identifiers. Common examples
    /// include `X`, `Y`, `R` and `I` as described above, as well as `W` for
    /// peak width and `M` for multiplicity.
    ///
    /// See [`FormatParser`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the format specifier cannot be parsed, or if the
    /// child parser for the respective block kind returns a fatal error.
    fn ambiguous_block(&mut self) -> Result<ChildParserExit> {
        let mut format_parser = FormatParser::from(self.lexer.clone());
        let format = format_parser.parse_format()?;

        match format.line_layout {
            LineLayout::RepeatingValue {
                incrementing,
                repeating,
            } => {
                let mut decoder = Decoder::from(format_parser.into_lexer());
                if let Some(kind) = format.kind {
                    decoder.set_title(kind);
                }
                decoder.set_incrementing(incrementing);
                decoder.set_repeating(repeating);
                let decoded_block = decoder.decode_source()?;
                self.lexer = decoder.into_lexer().morph();
                self.dataset
                    .errors
                    .extend(decoded_block.errors.into_iter().map(Error::from));
                self.dataset.data_tables.push(decoded_block.table);

                Ok(decoded_block.exit)
            }
            LineLayout::GroupedValues(identifiers) => {
                let mut table_parser =
                    TableParser::from(format_parser.into_lexer()).with_identifiers(identifiers);
                if let Some(kind) = format.kind {
                    table_parser.set_title(kind);
                }
                let tabulated_block = table_parser.tabulate_source()?;
                self.lexer = table_parser.into_lexer().morph();
                self.dataset.errors.extend(
                    tabulated_block
                        .errors
                        .into_iter()
                        .map(Error::from),
                );
                self.dataset
                    .data_tables
                    .push(tabulated_block.table);

                Ok(tabulated_block.exit)
            }
        }
    }
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
                assert!(parsed.is_ok());
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
