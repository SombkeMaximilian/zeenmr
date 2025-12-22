use logos::{Lexer, Logos};
use std::collections::HashMap;

/// JCAMP-DX header lexer.
#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern whitespace = r"[ \t]|(?&newline)")]
#[logos(skip r"(?&whitespace)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
#[logos(subpattern page = r"##PAGE[^\r\n]*")]
#[logos(skip r"(?&page)")]
enum HeaderToken {
    /// JCAMP-DX header keys start with `##` or `##.`. Bruker-specific keys
    /// start with `##$` but follow the same rules otherwise.
    #[token("##")]
    #[token("##.")]
    #[token("##$")]
    Key,
    /// The separator between key and value is always an equals sign.
    #[token("=")]
    Equals,
    /// Commas separating multiple values.
    #[token(",")]
    Comma,
    /// Opening parenthesis to start compound values or ranges.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end compound values or ranges.
    #[token(")")]
    CloseParenthesis,
    /// Starting angle brackets for strings with whitespace.
    #[token("<")]
    OpenAngle,
    /// Ending angle brackets for strings with whitespace.
    #[token(">")]
    CloseAngle,
    /// Integer or floating point values in standard or scientific notation.
    #[regex(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")]
    Numeric,
    /// Anything not numeric is a string.
    #[regex(r"[^ \t\r\n=#$,<>\(\)]*")]
    String,
    /// Data block for spectral data.
    #[regex(r"##(XYDATA)=", data_block)]
    #[regex(r"##(XYPOINTS)=", data_block)]
    #[regex(r"##(PEAK[\s_-]TABLE)=", data_block)]
    #[regex(r"##(PEAK[\s_-]ASSIGNMENTS)=", data_block)]
    #[regex(r"##(DATA[\s_-]TABLE)=", data_block)]
    DataBlock(DataBlock),
    /// End of a dataset.
    #[token("##END=")]
    End,
}

#[derive(Clone, PartialEq, Debug)]
enum DataBlock {
    Encoded {
        format: Vec<FormatToken>,
        data: Vec<EncodedToken>,
    },
    Grouped {
        kind: DataKind,
        format: Vec<FormatToken>,
        data: Vec<DataToken>,
    },
}

#[derive(Clone, Eq, PartialEq, Debug)]
enum DataKind {
    XYData,
    XYPoints,
    PeakTable,
    PeakAssignments,
}

fn data_block(lexer: &mut Lexer<HeaderToken>) -> DataBlock {
    let mut kind = &lexer.slice()[2..lexer.slice().len() - 1];
    let mut format_lexer = lexer.clone().morph::<FormatToken>();
    let mut format = Vec::new();
    while let Some(token) = format_lexer.next() {
        match token {
            Ok(FormatToken::DataBlockKind) => {
                kind = &format_lexer.slice();
            }
            Ok(FormatToken::End) => break,
            Ok(token) => format.push(token),
            Err(e) => panic!("lexing error: {:?}", e),
        }
    }
    let kind = kind
        .trim()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .to_uppercase();
    let kind = match kind.as_str() {
        "XYDATA" => DataKind::XYData,
        "XYPOINTS" => DataKind::XYPoints,
        "PEAKTABLE" | "PEAKS" => DataKind::PeakTable,
        "PEAKASSIGNMENTS" => DataKind::PeakAssignments,
        _ => panic!("unsupported data kind"),
    };
    match kind {
        DataKind::XYData => {
            let mut data_lexer = format_lexer.clone().morph::<EncodedToken>();
            let mut data = Vec::new();
            while let Some(token) = data_lexer.next() {
                match token {
                    Ok(EncodedToken::End) => break,
                    Ok(token) => data.push(token),
                    Err(e) => panic!("lexing error: {:?}", e),
                }
            }
            *lexer = data_lexer.morph();

            DataBlock::Encoded { format, data }
        }
        DataKind::XYPoints | DataKind::PeakTable | DataKind::PeakAssignments => {
            let mut data_lexer = format_lexer.clone().morph::<DataToken>();
            let mut data = Vec::new();
            while let Some(token) = data_lexer.next() {
                match token {
                    Ok(DataToken::End) => break,
                    Ok(token) => data.push(token),
                    Err(e) => panic!("lexing error: {:?}", e),
                }
            }
            *lexer = data_lexer.morph();

            DataBlock::Grouped { kind, format, data }
        }
    }
}

#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern parentheses = r"[()]")]
#[logos(skip r"(?&parentheses)")]
#[logos(subpattern comma = r",")]
#[logos(skip r"(?&comma)")]
enum FormatToken {
    /// Identifier for a quantity.
    ///
    /// Commonly used in the JCAMP-DX standard:
    /// - `X` for abscissa
    /// - `Y`, `R`, `I` for ordinates (any, real, imaginary)
    /// - `Z` for a third dimension
    /// - `W` for peak width
    /// - `M` for multiplicity
    /// - `A` for assignment
    #[regex(r"[A-Za-z][0-9]*", |lexer| lexer.slice().to_string())]
    Identifier(String),
    /// Abscissa is incremented by the number of values in a line.
    #[token("++")]
    Increment,
    /// Repeat current identifier until line ends.
    #[token("..")]
    Repeat,
    /// A `DATA TABLE` data block contains the exact type of data after a comma.
    #[regex(r"XYDATA|XYPOINTS|PEAKS|PEAK[\s_-]TABLE")]
    DataBlockKind,
    /// A new line ends a data block format specifier.
    #[regex(r"(?&newline)")]
    End,
}

#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
enum DataToken {
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)")]
    CheckPoints,
    /// Semicolons separate groups.
    #[token(";")]
    SemiColon,
    /// Commas separate members of groups.
    #[token(",")]
    Comma,
    /// Opening parenthesis to start a group.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end a group.
    #[token(")")]
    CloseParenthesis,
    /// Starting angle brackets for strings with whitespace.
    #[token("<")]
    OpenAngle,
    /// Ending angle brackets for strings with whitespace.
    #[token(">")]
    CloseAngle,
    /// Numeric values.
    #[regex(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?", |lexer| lexer.slice().parse::<f64>().unwrap())]
    Numeric(f64),
    /// Anything else is a string.
    #[regex(r"[^ \t\r\n?,;<>\(\)]*", |lexer| lexer.slice().to_string())]
    String(String),
    /// Invalid data point, marked as `?` in the JCAMP-DX standard.
    #[token("?")]
    Invalid,
    /// End of data block.
    #[regex(r"##PAGE=[^\r\n]*")]
    #[regex(r"##END[^\r\n]*")]
    End,
}

#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
enum EncodedToken {
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)")]
    CheckPoint,
    /// Numeric values, `AFFN` in the JCAMP-DX standard.
    #[regex(r"[+-]?(0|[1-9]\d*)(\.\d+)?([eE][+-]\d+)?", affn)]
    Numeric(i64),
    /// Compressed value, `SQZ` in the JCAMP-DX standard.
    #[regex(r"[@A-Ia-i]\d*", asdf)]
    Compressed(i64),
    /// Difference value, `DIF` in the JCAMP-DX standard.
    #[regex(r"[%J-Rj-r]\d*", asdf)]
    Difference(i64),
    /// Duplicate previous value, `DUP` in the JCAMP-DX standard.
    #[regex(r"[S-Zs]\d*", |lexer| asdf(lexer) as u32)]
    Duplicate(u32),
    /// Invalid data point, marked as `?` in the JCAMP-DX standard.
    #[token("?")]
    Invalid,
    /// End of data block.
    #[regex(r"##PAGE=[^\r\n]*")]
    #[regex(r"##END[^\r\n]*")]
    End,
}

/// Parse an `AFFN` numeric value.
fn affn(lexer: &Lexer<EncodedToken>) -> i64 {
    match lexer.slice().parse::<i64>() {
        Ok(int) => int,
        Err(_) => lexer.slice().parse::<f64>().unwrap() as i64,
    }
}

/// Parse an `ASDF` compressed, difference, or duplicate value.
fn asdf(lexer: &Lexer<EncodedToken>) -> i64 {
    let encoded = &lexer.slice()[..1];
    let decoded: i64 = match encoded {
        "@" | "%" => 0,
        "A" | "a" | "J" | "j" | "S" => 1,
        "B" | "b" | "K" | "k" | "T" => 2,
        "C" | "c" | "L" | "l" | "U" => 3,
        "D" | "d" | "M" | "m" | "V" => 4,
        "E" | "e" | "N" | "n" | "W" => 5,
        "F" | "f" | "O" | "o" | "X" => 6,
        "G" | "g" | "P" | "p" | "Y" => 7,
        "H" | "h" | "Q" | "q" | "Z" => 8,
        "I" | "i" | "R" | "r" | "s" => 9,
        _ => unreachable!("invalid ASDF character: {}", encoded),
    };
    let sign = match encoded.chars().next().unwrap() {
        '@' | '%' | 'A'..='Z' | 's' => 1,
        'a'..='r' => -1,
        _ => unreachable!("invalid ASDF character: {}", encoded),
    };
    let numeric = &lexer.slice()[1..];
    let order = numeric.len() as u32;
    let numeric = if order > 0 {
        numeric.parse::<i64>().unwrap()
    } else {
        0
    };

    sign * (decoded * 10_i64.pow(order) + numeric)
}

/// Possible values in JCAMP-DX files.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    /// Empty values for unset parameters.
    Empty,
    /// Integer values.
    Integer(i64),
    /// Float values.
    Float(f64),
    /// Everything else is treated as a string.
    String(String),
    /// Arrays of values.
    Array(Vec<Value>),
}

/// Parser implementation for the JCAMP-DX file format.
#[derive(Debug)]
pub(crate) struct Parser<'source> {
    /// Lexer for tokenizing the input source.
    lexer: Lexer<'source, HeaderToken>,
    /// Parsed parameters.
    parameters: HashMap<String, Value>,
    /// Current key being processed.
    current_key: &'source str,
    /// Current value being built.
    current_value: Value,
    /// Bounded structure stack for values inside parentheses.
    bounded_stack: Vec<Vec<Value>>,
    /// Strings not separated by a comma are automatically combined.
    auto_concatenate: bool,
}

impl<'source> From<&'source str> for Parser<'source> {
    fn from(value: &'source str) -> Self {
        let mut lexer = Lexer::new(value);
        while let Some(token) = lexer.next() {
            match token {
                Ok(HeaderToken::Key) => break,
                Ok(_) => continue,
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }
        let start = lexer.span().end;
        while let Some(token) = lexer.next() {
            match token {
                Ok(HeaderToken::Key) => panic!("unexpected subsequent key tokens"),
                Ok(HeaderToken::Equals) => break,
                Ok(HeaderToken::DataBlock(_)) => panic!("unexpected data block while parsing key"),
                Ok(HeaderToken::End) => panic!("unexpected end of file after first key"),
                Ok(_) => continue,
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }
        let end = lexer.span().start;

        Self {
            lexer,
            parameters: HashMap::new(),
            current_key: &value[start..end],
            current_value: Value::Empty,
            bounded_stack: Vec::new(),
            auto_concatenate: false,
        }
    }
}

impl<'source> Parser<'source> {
    pub(crate) fn parse_source(mut self) {
        while let Some(token) = self.lexer.next() {
            let clear_auto_concatenate = token != Ok(HeaderToken::Comma);
            match token {
                Ok(HeaderToken::Key) => self.key(),
                Ok(HeaderToken::Equals) => panic!("unexpected key value separator"),
                Ok(HeaderToken::Comma) => self.comma(),
                Ok(HeaderToken::OpenParenthesis) | Ok(HeaderToken::OpenAngle) => {
                    self.start_bounded()
                }
                Ok(HeaderToken::CloseParenthesis) | Ok(HeaderToken::CloseAngle) => {
                    self.end_bounded()
                }
                Ok(HeaderToken::Numeric) => self.numeric(),
                Ok(HeaderToken::String) => self.string(),
                Ok(HeaderToken::DataBlock(data_block)) => self.data_block(data_block),
                Ok(HeaderToken::End) => {
                    self.parameters
                        .insert(self.current_key.to_string(), self.current_value.clone());
                    break;
                }
                Err(e) => panic!("lexing error: {:?}", e),
            }
            if clear_auto_concatenate {
                self.auto_concatenate = false;
            }
        }
    }

    fn is_bounded(&self) -> bool {
        !self.bounded_stack.is_empty()
    }

    fn key(&mut self) {
        let old = std::mem::replace(&mut self.current_value, Value::Empty);
        self.parameters
            .insert(self.current_key.to_string(), old);
        let start = self.lexer.span().end;
        while let Some(token) = self.lexer.next() {
            match token {
                Ok(HeaderToken::Key) => panic!("unexpected subsequent key tokens"),
                Ok(HeaderToken::Equals) => break,
                Ok(HeaderToken::DataBlock(_)) => panic!("unexpected data block while parsing key"),
                Ok(HeaderToken::End) => panic!("unexpected end of file after key"),
                Ok(_) => continue,
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }
        let end = self.lexer.span().start;
        self.current_key = &self.lexer.source()[start..end];
    }

    fn comma(&mut self) {
        if !self.auto_concatenate {
            self.auto_concatenate = true;
        } else {
            if let Some(bounded) = self.bounded_stack.last_mut() {
                bounded.push(Value::Empty);
            } else {
                match self.current_value {
                    Value::Empty => {
                        self.current_value = Value::Array(vec![Value::Empty]);
                    }
                    Value::Array(ref mut array) => {
                        array.push(Value::Empty);
                    }
                    _ => {
                        let old = std::mem::replace(&mut self.current_value, Value::Empty);
                        self.current_value = Value::Array(vec![old, Value::Empty]);
                    }
                }
            }
        }
    }

    fn start_bounded(&mut self) {
        self.bounded_stack.push(Vec::new());
    }

    fn end_bounded(&mut self) {
        let value = match self.bounded_stack.pop() {
            Some(closed) => match closed.len() {
                0 => Value::Empty,
                1 => closed.into_iter().next().unwrap(),
                _ => Value::Array(closed),
            },
            None => panic!("unmatched closing parenthesis"),
        };
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(value);
        } else {
            match self.current_value {
                Value::Empty => {
                    self.current_value = value;
                }
                Value::Array(ref mut array) => {
                    array.push(value);
                }
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value = Value::Array(vec![old, value]);
                }
            }
        }
    }

    fn numeric(&mut self) {
        let value = match self.lexer.slice().parse::<i64>() {
            Ok(int) => Value::Integer(int),
            Err(_) => Value::Float(self.lexer.slice().parse::<f64>().unwrap()),
        };
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(value);
        } else {
            match self.current_value {
                Value::Empty => {
                    self.current_value = value;
                }
                Value::Array(ref mut array) => {
                    array.push(value);
                }
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value = Value::Array(vec![old, value]);
                }
            }
        }
    }

    fn string(&mut self) {
        let value = self.lexer.slice();
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(Value::String(value.to_string()));
        } else {
            match self.current_value {
                Value::Empty => {
                    self.current_value = Value::String(value.to_string());
                }
                Value::String(ref mut previous) => {
                    if self.auto_concatenate {
                        let old = std::mem::replace(&mut self.current_value, Value::Empty);
                        self.current_value =
                            Value::Array(vec![old, Value::String(value.to_string())]);
                    } else {
                        if previous.len() > 0 {
                            previous.push(' ');
                        }
                        previous.push_str(value);
                    }
                }
                Value::Array(ref mut array) => match array.last_mut() {
                    Some(&mut Value::String(ref mut previous)) => {
                        if previous.len() > 0 {
                            previous.push(' ');
                        }
                        previous.push_str(value);
                    }
                    _ => array.push(Value::String(value.to_string())),
                },
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value = Value::Array(vec![old, Value::String(value.to_string())]);
                }
            }
        }
    }

    fn data_block(&mut self, data_block: DataBlock) {
        match data_block {
            DataBlock::Encoded { format, data } => match &format[..] {
                [
                    FormatToken::Identifier(x),
                    FormatToken::Increment,
                    FormatToken::Identifier(y),
                    FormatToken::Repeat,
                    FormatToken::Identifier(y_check),
                ] => {
                    if y != y_check {
                        panic!("non-repeating identifier");
                    } else {
                        self.encoded_block(data);
                    }
                }
                _ => panic!("incompatible format"),
            },
            _ => todo!(),
        }
    }

    fn encoded_block(&mut self, data: Vec<EncodedToken>) {
        let mut check_point = true;
        let mut last_was_difference = None;
        let mut integrity_check = false;
        let mut decoded = Vec::new();
        let mut check_points = vec![0_usize];
        let mut check_point_values = Vec::new();
        let mut data_stream = data.into_iter();
        while let Some(token) = data_stream.next() {
            match token {
                EncodedToken::CheckPoint => {
                    check_point = true;
                    if last_was_difference.is_some() {
                        integrity_check = true;
                        check_points.push(decoded.len() - 1);
                    } else {
                        check_points.push(decoded.len());
                    }
                }
                EncodedToken::Numeric(value) | EncodedToken::Compressed(value) => {
                    if check_point {
                        check_point = false;
                        check_point_values.push(value);
                    } else {
                        if integrity_check {
                            let previous = *decoded.last().unwrap();
                            if !(value == previous) {
                                panic!("integrity check failed");
                            }
                        } else {
                            decoded.push(value);
                        }
                        last_was_difference = None;
                        integrity_check = false;
                    }
                }
                EncodedToken::Difference(value) => {
                    if check_point {
                        panic!("first value after check point cannot be difference");
                    }
                    let result = *decoded.last().unwrap() + value;
                    decoded.push(result);
                    last_was_difference = Some(value);
                }
                EncodedToken::Duplicate(value) => {
                    if check_point {
                        panic!("first value after check point cannot be duplicate");
                    }
                    let previous = *decoded.last().unwrap();
                    if let Some(diff) = last_was_difference {
                        let values = (1..value).map(|i| previous + (diff * i as i64));
                        decoded.extend(values);
                    } else {
                        for _ in 1..value {
                            decoded.push(previous);
                        }
                    }
                }
                EncodedToken::Invalid => panic!("invalid value"),
                EncodedToken::End => break,
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

    macro_rules! lexer_test {
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
                let mut lexer = HeaderToken::lexer(&content);
                let mut data_blocks = 0;
                while let Some(token) = lexer.next() {
                    assert!(token.is_ok());
                    if let Ok(HeaderToken::DataBlock(data_block)) = token {
                        match data_block {
                            DataBlock::Encoded { format, data } => {
                                // lower bound for 131072 points + around 7000 checkpoints
                                assert!(data.len() > 2_usize.pow(17) + 14_000);
                                data_blocks += 1;
                            }
                            DataBlock::Grouped { format, data, .. } => {
                                panic!("unexpected grouped data block")
                            }
                        }
                    }
                }
                assert_ne!(data_blocks, 0);
            }
        };
    }

    lexer_test!(lexer_v5_ntuples_affn, "v5", "ntuples_affn.dx");
    lexer_test!(lexer_v5_ntuples_pac, "v5", "ntuples_pac.dx");
    lexer_test!(lexer_v5_ntuples_sqz, "v5", "ntuples_sqz.dx");
    lexer_test!(lexer_v5_ntuples_difdup, "v5", "ntuples_difdup.dx");
    lexer_test!(lexer_v5_xydata_affn, "v5", "xydata_affn.dx");
    lexer_test!(lexer_v5_xydata_pac, "v5", "xydata_pac.dx");
    lexer_test!(lexer_v5_xydata_sqz, "v5", "xydata_sqz.dx");
    lexer_test!(lexer_v5_xydata_difdup, "v5", "xydata_difdup.dx");
    lexer_test!(lexer_v6_ntuples_affn, "v6", "ntuples_affn.dx");
    lexer_test!(lexer_v6_ntuples_pac, "v6", "ntuples_pac.dx");
    lexer_test!(lexer_v6_ntuples_sqz, "v6", "ntuples_sqz.dx");
    lexer_test!(lexer_v6_ntuples_difdup, "v6", "ntuples_difdup.dx");
    lexer_test!(lexer_v6_xydata_affn, "v6", "xydata_affn.dx");
    lexer_test!(lexer_v6_xydata_pac, "v6", "xydata_pac.dx");
    lexer_test!(lexer_v6_xydata_sqz, "v6", "xydata_sqz.dx");
    lexer_test!(lexer_v6_xydata_difdup, "v6", "xydata_difdup.dx");

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
