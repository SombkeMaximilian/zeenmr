use logos::{Lexer, Logos};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Bruker parameter file lexer.
#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern whitespace = r"[ \t]|(?&newline)")]
#[logos(skip r"(?&whitespace)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
enum Token {
    /// JCAMP-DX keys start with `##`. Bruker-specific keys start with `##$`.
    #[token("##")]
    #[token("##$")]
    Key,
    /// The separator between key and value is always an equals sign.
    #[token("=")]
    Equals,
    /// Opening parenthesis to start compound values or ranges.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end compound values or ranges.
    #[token(")")]
    CloseParenthesis,
    /// Starting angle brackets for specific values.
    #[token("<")]
    OpenAngle,
    /// Ending angle brackets for specific values.
    #[token(">")]
    CloseAngle,
    /// Ranges specify how many values are in a following array.
    #[regex(r"\(\d+\.\.\d+\)")]
    Range,
    /// Numeric values.
    #[regex(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")]
    Numeric,
    /// Version values that can be disambiguated from [`Token::Numeric`]
    /// immediately by their three-part structure.
    #[regex(r"\d+\.\d+\.\d+")]
    Version,
    /// String values.
    #[regex(r"[^ \t\r\n=#$<>\(\)]*")]
    String,
    /// The end of a parameter file is marked with `##END=`.
    #[token("##END=")]
    End,
}

/// Possible values in Bruker parameter files.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(untagged))]
pub enum Value {
    /// Empty values for unset parameters.
    Empty,
    /// Integer values.
    Integer(i64),
    /// Float values.
    Float(f64),
    /// Version values with two or three parts.
    Version(u32, u32, Option<u32>),
    /// Everything else is treated as a string.
    String(String),
    /// Arrays of values.
    Array(Vec<Value>),
}

/// Parser implementation for Bruker parameter files.
pub(crate) struct Parser<'source> {
    /// Lexer for tokenizing the input source.
    lexer: Lexer<'source, Token>,
    /// Parsed parameters.
    parameters: HashMap<String, Value>,
    /// Current key being processed.
    current_key: &'source str,
    /// Current value being built.
    current_value: Value,
    /// Bounded structure stack.
    bounded_stack: Vec<Vec<Value>>,
    /// Expected size of the current array, if known.
    array_size: Option<usize>,
}

impl<'source> From<&'source str> for Parser<'source> {
    fn from(value: &'source str) -> Self {
        let mut lexer = Lexer::new(value);
        while let Some(token) = lexer.next() {
            match token {
                Ok(Token::Key) => break,
                Ok(_) => continue,
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }
        let current_key = match lexer.next() {
            Some(Ok(Token::String)) => lexer.slice(),
            Some(Ok(_)) => panic!("expected string after key"),
            Some(Err(e)) => panic!("lexing error: {:?}", e),
            None => panic!("unexpected end of input after key"),
        };
        match lexer.next() {
            Some(Ok(Token::Equals)) => {}
            Some(Ok(_)) => panic!("expected equals sign after first key"),
            Some(Err(e)) => panic!("lexing error: {:?}", e),
            None => panic!("unexpected end of input after first key"),
        }

        Self {
            lexer,
            parameters: HashMap::new(),
            current_key,
            current_value: Value::Empty,
            bounded_stack: Vec::new(),
            array_size: None,
        }
    }
}

impl<'source> Parser<'source> {
    /// Parse a Bruker parameter file into a map of keys and values.
    pub(crate) fn parse_source(mut self) -> HashMap<String, Value> {
        while let Some(token) = self.lexer.next() {
            match token {
                Ok(Token::Key) => self.key(),
                Ok(Token::Equals) => panic!("unexpected equals sign"),
                Ok(Token::OpenParenthesis) | Ok(Token::OpenAngle) => self.start_bounded(),
                Ok(Token::CloseParenthesis) | Ok(Token::CloseAngle) => self.end_bounded(),
                Ok(Token::Range) => self.range(),
                Ok(Token::Numeric) => self.numeric(),
                Ok(Token::Version) => self.version(),
                Ok(Token::String) => self.string(),
                Ok(Token::End) => {
                    self.parameters
                        .insert(self.current_key.to_string(), self.current_value);
                    break;
                }
                Err(e) => panic!("lexing error: {:?}", e),
            }
        }

        self.parameters
    }

    /// Check if currently inside a bounded structure.
    #[inline]
    fn is_bounded(&self) -> bool {
        !self.bounded_stack.is_empty()
    }

    /// Check if currently building an array.
    #[inline]
    fn is_array(&self) -> bool {
        self.array_size.is_some()
    }

    /// Process a key token.
    #[inline]
    fn key(&mut self) {
        if self.is_bounded() || self.is_array() {
            panic!("unexpected key inside bounded structure or array");
        }
        let old = std::mem::replace(&mut self.current_value, Value::Empty);
        self.parameters
            .insert(self.current_key.to_string(), old);
        self.current_key = match self.lexer.next() {
            Some(Ok(Token::String)) => self.lexer.slice(),
            Some(Ok(_)) => panic!("expected string after key"),
            Some(Err(e)) => panic!("lexing error: {:?}", e),
            None => panic!("unexpected end of input after key"),
        };
        match self.lexer.next() {
            Some(Ok(Token::Equals)) => {}
            Some(Ok(_)) => panic!("expected equals sign after key"),
            Some(Err(e)) => panic!("lexing error: {:?}", e),
            None => panic!("unexpected end of input after key"),
        }
    }

    /// Start a new bounded structure.
    #[inline]
    fn start_bounded(&mut self) {
        self.bounded_stack.push(Vec::new());
    }

    /// End the current bounded structure.
    #[inline]
    fn end_bounded(&mut self) {
        let value = match self.bounded_stack.pop() {
            Some(closed) => match closed.len() {
                0 => Value::Empty,
                1 => closed.into_iter().next().unwrap(),
                _ => Value::Array(closed),
            },
            None => panic!("unmatched closing parenthesis or angle bracket"),
        };
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(value);
            if let Some(array_size) = self.array_size {
                if bounded.len() == array_size {
                    self.array_size = None;
                } else if bounded.len() >= array_size {
                    panic!("too many items in array");
                }
            }
        } else if let Some(array_size) = self.array_size {
            match self.current_value {
                Value::Array(ref mut array) => {
                    array.push(value);
                    if array.len() == array_size {
                        self.array_size = None;
                    } else if array.len() >= array_size {
                        panic!("too many items in array");
                    }
                }
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value = Value::Array(vec![old, value]);
                }
            }
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

    /// Parse a range token to determine the size of the following array.
    #[inline]
    fn range(&mut self) {
        static RANGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)\.\.(\d+)").unwrap());
        if !self.bounded_stack.is_empty() {
            panic!("unexpected range inside bounded structure");
        }
        // unwraps are safe since range tokens are of the form (start..end)
        let captures = RANGE.captures(self.lexer.slice()).unwrap();
        let start = captures
            .get(1)
            .unwrap()
            .as_str()
            .parse::<f64>()
            .unwrap();
        let end = captures
            .get(2)
            .unwrap()
            .as_str()
            .parse::<f64>()
            .unwrap();
        let array_size = (end - start + 1.0) as usize;
        self.array_size = Some(array_size);
        self.current_value = Value::Array(Vec::with_capacity(array_size));
    }

    /// Parse a numeric token.
    #[inline]
    fn numeric(&mut self) {
        let value = match self.lexer.slice().parse::<i64>() {
            Ok(int) => Value::Integer(int),
            Err(_) => Value::Float(self.lexer.slice().parse::<f64>().unwrap()),
        };
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(value);
        } else if let Some(array_size) = self.array_size {
            match self.current_value {
                Value::Array(ref mut array) => {
                    array.push(value);
                    if array.len() == array_size {
                        self.array_size = None;
                    } else if array.len() >= array_size {
                        panic!("too many items in array");
                    }
                }
                _ => unreachable!("array_size is set but current_value is not an array"),
            }
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
            };
        }
    }

    /// Parse a version token.
    #[inline]
    fn version(&mut self) {
        static VERSION: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(\d+)\.(\d+)\.(\d+)").unwrap());
        let caps = VERSION.captures(self.lexer.slice()).unwrap();
        let major = caps
            .get(1)
            .unwrap()
            .as_str()
            .parse::<u32>()
            .unwrap();
        let minor = caps
            .get(2)
            .unwrap()
            .as_str()
            .parse::<u32>()
            .unwrap();
        let patch = caps
            .get(3)
            .unwrap()
            .as_str()
            .parse::<u32>()
            .unwrap();
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(Value::Version(major, minor, Some(patch)));
        } else if let Some(array_size) = self.array_size {
            match self.current_value {
                Value::Array(ref mut array) => {
                    array.push(Value::Version(major, minor, Some(patch)));
                    if array.len() == array_size {
                        self.array_size = None;
                    } else if array.len() >= array_size {
                        panic!("too many items in array");
                    }
                }
                _ => unreachable!("array_size is set but current_value is not an array"),
            }
        } else {
            match self.current_value {
                Value::Empty => {
                    self.current_value = Value::Version(major, minor, Some(patch));
                }
                Value::Array(ref mut array) => {
                    array.push(Value::Version(major, minor, Some(patch)));
                }
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value =
                        Value::Array(vec![old, Value::Version(major, minor, Some(patch))]);
                }
            }
        }
    }

    /// Parse a string token.
    #[inline]
    fn string(&mut self) {
        let value = self.lexer.slice();
        if let Some(bounded) = self.bounded_stack.last_mut() {
            bounded.push(Value::String(value.to_string()));
        } else if let Some(array_size) = self.array_size {
            match self.current_value {
                Value::Array(ref mut array) => {
                    array.push(Value::String(value.to_string()));
                    if array.len() == array_size {
                        self.array_size = None;
                    } else if array.len() >= array_size {
                        panic!("too many items in array");
                    }
                }
                _ => unreachable!("array_size is set but current_value is not an array"),
            }
        } else {
            match self.current_value {
                Value::Empty => {
                    self.current_value = Value::String(value.to_string());
                }
                Value::String(ref mut previous) => {
                    if previous.len() > 0 {
                        previous.push(' ')
                    }
                    previous.push_str(value);
                }
                Value::Array(ref mut array) => {
                    if let Some(array_size) = self.array_size {
                        if array.len() == array_size - 1 {
                            self.array_size = None;
                        } else if array.len() >= array_size {
                            panic!("too many items in array");
                        }
                    }
                    match array.last_mut() {
                        Some(&mut Value::String(ref mut last)) => {
                            if last.len() > 0 {
                                last.push(' ');
                            }
                            last.push_str(value);
                        }
                        _ => array.push(Value::String(value.to_string())),
                    }
                }
                _ => {
                    let old = std::mem::replace(&mut self.current_value, Value::Empty);
                    self.current_value = Value::Array(vec![old, Value::String(value.to_string())]);
                }
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

    #[test]
    fn lexer() {
        let acqus_path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01")
            .join("10")
            .join("acqus");
        let acqus = read_to_string(acqus_path).unwrap();
        let mut lexer = Token::lexer(&acqus);
        let mut end = false;
        while let Some(token) = lexer.next() {
            assert!(token.is_ok());
            if let Ok(Token::End) = token {
                end = true;
            }
        }
        assert!(end);
    }

    #[test]
    fn parser() {
        let acqus_path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01")
            .join("10")
            .join("acqus");
        let acqus = read_to_string(acqus_path).unwrap();
        let acqus = Parser::from(acqus.as_str()).parse_source();
        let important = ["BYTORDA", "DTYPA", "NC", "NUC1", "SFO1", "SW", "SW_h", "TD"];
        let expected = [
            Value::Integer(0),
            Value::Integer(0),
            Value::Integer(-6),
            Value::String("1H".to_string()),
            Value::Float(600.252_821_089_118),
            Value::Float(20.023_613_962_234_7),
            Value::Float(12019.230_769_230_8),
            Value::Integer(65536),
        ];
        for (key, expected) in important.iter().zip(expected.iter()) {
            assert_eq!(acqus.get(*key).unwrap(), expected);
        }
        let procs_path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01")
            .join("10")
            .join("pdata")
            .join("10")
            .join("procs");
        let procs = read_to_string(procs_path).unwrap();
        let procs = Parser::from(procs.as_str()).parse_source();
        let important = ["BYTORDP", "DTYPP", "NC_proc", "SI", "SPECTYP", "SW_p"];
        let expected = [
            Value::Integer(0),
            Value::Integer(0),
            Value::Integer(0),
            Value::Integer(131072),
            Value::String("PROTON".to_string()),
            Value::Float(12019.230_769_230_8),
        ];
        for (key, expected) in important.iter().zip(expected.iter()) {
            assert_eq!(procs.get(*key).unwrap(), expected);
        }
    }
}
