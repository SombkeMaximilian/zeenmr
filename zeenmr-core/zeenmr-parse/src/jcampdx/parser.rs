use logos::{Lexer, Logos};
use std::collections::HashMap;

/// JCAMP-DX header lexer.
#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern whitespace = r"[ \t]|(?&newline)")]
#[logos(skip r"(?&whitespace)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
enum HeaderToken {
    /// JCAMP-DX header keys start with `##` or `##.`.
    #[token("##")]
    #[token("##.")]
    Key,
    /// The separator between key and value is always an equals sign.
    #[token("=")]
    Equals,
    /// Commas separating multiple values.
    #[token(",")]
    Comma,
    /// Integer or floating point values in standard or scientific notation.
    #[regex(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")]
    Numeric,
    /// Anything not numeric is a string.
    #[regex(r"[^ \t\r\n=#$,]*")]
    String,
    /// Data block for spectral data.
    #[regex(r"##(XYDATA)=", data_block)]
    #[regex(r"##(XYPOINTS)=", data_block)]
    #[regex(r"##(PEAK[\s_-]TABLE)=", data_block)]
    #[regex(r"##(DATA[\s_-]TABLE)=", data_block)]
    DataBlock((DataKind, Vec<FormatToken>, Vec<DataToken>)),
    /// End of a dataset.
    #[token("##END=")]
    End,
}

#[derive(Clone, Eq, PartialEq, Debug)]
enum DataKind {
    XYData,
    XYPoints,
    PeakTable,
}

fn data_block(lexer: &mut Lexer<HeaderToken>) -> (DataKind, Vec<FormatToken>, Vec<DataToken>) {
    let mut kind = &lexer.slice()[2..lexer.slice().len() - 1];
    let mut format_lexer = lexer.clone().morph::<FormatToken>();
    let mut format = Vec::new();
    while let Some(token) = format_lexer.next() {
        match token {
            Ok(FormatToken::Comma) => continue,
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
        _ => panic!("unsupported data kind"),
    };
    while format[0] == FormatToken::OpenParenthesis
        && format[format.len() - 1] == FormatToken::CloseParenthesis
    {
        format.remove(0);
        format.remove(format.len() - 1);
    }
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

    (kind, format, data)
}

#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
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
    /// Opening parenthesis to begin pattern.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end pattern.
    #[token(")")]
    CloseParenthesis,
    /// Extra information after a comma.
    #[token(",")]
    Comma,
    /// A `DATA TABLE` data block contains the exact type of data after a comma.
    #[regex(r"XYDATA|XYPOINTS|PEAK[\s_-]TABLE|PEAK[\s_-]ASSIGNMENTS")]
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
    /// For XY pairs, XYZ triplets, etc., semicolons are to separate groups.
    #[token(";")]
    GroupSeparator,
    /// Within XY pairs, XYZ triplets, etc., commas are used to separate
    /// coordinates.
    #[token(",")]
    CoordinateSeparator,
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)")]
    CheckPoint,
    /// Numeric values, `AFFN` in the JCAMP-DX standard.
    #[regex(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", affn)]
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
fn affn(lexer: &Lexer<DataToken>) -> i64 {
    match lexer.slice().parse::<i64>() {
        Ok(int) => int,
        Err(_) => lexer.slice().parse::<f64>().unwrap() as i64,
    }
}

/// Parse an `ASDF` compressed, difference, or duplicate value.
fn asdf(lexer: &Lexer<DataToken>) -> i64 {
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

/// Data block containing either real or imaginary values.
#[derive(Clone, PartialEq, Debug)]
pub enum DataBlock {
    Real(Vec<f64>),
    Imag(Vec<f64>),
}

/// Parser implementation for the JCAMP-DX file format.
#[derive(Debug)]
pub(crate) struct Parser<'source> {
    /// Lexer for tokenizing the input source.
    lexer: Lexer<'source, HeaderToken>,
    /// Parsed parameters.
    parameters: HashMap<String, Value>,
    /// Parsed data blocks.
    data_blocks: Vec<DataBlock>,
    /// Current key being processed.
    current_key: &'source str,
    /// Current value being built.
    current_value: Value,
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
            data_blocks: Vec::new(),
            current_key: &value[start..end],
            current_value: Value::Empty,
        }
    }
}

impl<'source> Parser<'source> {
    pub(crate) fn parse_source(&mut self) {
        while let Some(token) = self.lexer.next() {
            match token {
                Ok(HeaderToken::Key) => self.key(),
                Ok(HeaderToken::Equals) => panic!("unexpected key value separator"),
                Ok(HeaderToken::Comma) => continue,
                Ok(HeaderToken::String) => continue,
                Ok(HeaderToken::Numeric) => continue,
                Ok(HeaderToken::DataBlock(_)) => continue,
                Ok(HeaderToken::End) => break,
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }
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
        self.current_key = &self.current_key[start..end];
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
                    if let Ok(HeaderToken::DataBlock((_, _, data))) = token {
                        // lower bound for 131072 points + around 7000 checkpoints
                        data_blocks += 1;
                        assert!(data.len() > 2_usize.pow(17) + 14_000);
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

    #[test]
    fn current() {
        let text = "##DATA CLASS= XYDATA";
        let mut lexer = HeaderToken::lexer(&text);
        while let Some(token) = lexer.next() {
            println!("{:?}: {}", token, &lexer.slice());
        }
    }
}
