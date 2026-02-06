use crate::Location;
use crate::jcampdx::decoding::error::{Error, Result};
use crate::jcampdx::decoding::{DecodedBlock, DecodedBlockBuilder, EncodedToken};
use crate::location::Cursor;
use logos::{Lexer, Logos};
use std::num::{IntErrorKind, ParseIntError};

/// `AFFN` numeric values.
#[derive(Copy, Clone, PartialEq, Debug)]
enum Affn {
    /// Matches `[+-]?(0|[1-9]\d*)([eE][+-]\d+)?`.
    I64(i64),
    /// Matches `[+-]?(0|[1-9]\d*)(\.\d+)?([eE][+-]\d+)?`.
    F64(f64),
}

/// Decoding phase relative to checkpoints.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Phase {
    /// Expecting the checkpoint value.
    ///
    /// `DIF` and `DUP` encodings are not allowed in this phase.
    CheckPoint,
    /// Expecting the first data value after a checkpoint.
    ///
    /// `DIF` and `DUP` encodings are not allowed in this phase.
    FirstData,
    /// Normal data stream.
    Data,
}

/// State of the [`Decoder`].
///
/// Tracks how the previous value was produced which determines how subsequent
/// values are interpreted in some contexts. Independent of the current
/// [`Phase`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum State {
    /// Default state with no context.
    Normal,
    /// The last value was difference encoded.
    ///
    /// Subsequent `DUP` tokens are repeating this difference, not the resulting
    /// value. The contained value is the applied difference.
    LastWasDifference(i64),
    /// An integrity check is performed if the last value was `DIF` encoded and
    /// a checkpoint was encountered.
    ///
    ///
    /// The first data value after a checkpoint ([`Numeric`] or [`Compressed`]
    /// token) is compared against the contained value. If they are unequal, the
    /// integrity check fails.
    ///
    /// [`Numeric`]: EncodedToken::Numeric
    /// [`Compressed`]: EncodedToken::Compressed
    IntegrityCheck,
}

/// Decoder for the `ASDF` format used in JCAMP-DX files.
#[derive(Debug)]
pub(crate) struct Decoder<'source> {
    /// Lexer for the encoding format.
    lexer: Lexer<'source, EncodedToken>,
    /// Decoding phase relative to checkpoints.
    phase: Phase,
    /// Current state for duplicate values and integrity checks.
    state: State,
    /// Decoded block being constructed.
    builder: DecodedBlockBuilder,
}

impl<'source> From<&'source str> for Decoder<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: EncodedToken::lexer(value),
            phase: Phase::CheckPoint,
            state: State::Normal,
            builder: DecodedBlockBuilder::default(),
        }
    }
}

impl<'source, T> From<Lexer<'source, T>> for Decoder<'source>
where
    T: Logos<'source, Source = str> + Clone,
    T::Extras: Clone + Into<Cursor>,
{
    fn from(value: Lexer<'source, T>) -> Self {
        Self {
            lexer: value.morph(),
            phase: Phase::CheckPoint,
            state: State::Normal,
            builder: DecodedBlockBuilder::default(),
        }
    }
}

impl<'source> Decoder<'source> {
    /// Recovers the [`Lexer`] from the `Decoder`.
    pub(crate) fn into_lexer(mut self) -> Lexer<'source, EncodedToken> {
        self.lexer
    }

    pub(crate) fn decode_source(&mut self) -> Result<DecodedBlock> {
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                EncodedToken::CheckPoint => self.check_point(),
                EncodedToken::Numeric => self.numeric()?,
                EncodedToken::Compressed => self.compressed()?,
                EncodedToken::Difference => self.difference()?,
                EncodedToken::Duplicate => self.duplicate()?,
                EncodedToken::Invalid => self.invalid(),
                EncodedToken::End => break,
            }
        }

        Ok(std::mem::take(&mut self.builder).finalize())
    }

    fn check_point(&mut self) {
        match self.state {
            State::LastWasDifference(_) => {
                self.state = State::IntegrityCheck;
                self.builder.checkpoint_integrity_check();
            }
            _ => {
                self.builder.checkpoint();
            }
        }
        self.phase = Phase::CheckPoint;
    }

    fn numeric(&mut self) -> Result<()> {
        match self.phase {
            Phase::CheckPoint => {
                match self.parse_affn() {
                    Ok(Affn::I64(value)) => {
                        self.builder.push_checkpoint_value(value as f64);
                    }
                    Ok(Affn::F64(value)) => {
                        self.builder.push_checkpoint_value(value);
                    }
                    Err(e) => match e.kind() {
                        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                            self.builder.push_checkpoint_value(
                                self
                                    .lexer
                                    .slice()
                                    .parse::<f64>()
                                    .expect("lexer (regex) should not match unparsable numerics")
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                self.phase = Phase::FirstData;

                Ok(())
            }
            Phase::Data | Phase::FirstData => {
                match self.state {
                    State::Normal | State::LastWasDifference(_) => match self.parse_affn() {
                        Ok(Affn::I64(value)) => self.builder.push_decoded_i64(value),
                        Ok(Affn::F64(value)) => self.builder.push_decoded_f64(value),
                        Err(e) => match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                self.builder.push_error(Error::overflow_with_index(
                                    self.lexer.location(),
                                    self.builder.decoded_len(),
                                ));
                                self.builder.push_decoded_i64(i64::MIN);
                            }
                            _ => unreachable!(),
                        }
                    }
                    State::IntegrityCheck => match self.parse_affn() {
                        Ok(Affn::I64(value)) => self.integrity_check(value)?,
                        Ok(Affn::F64(_)) => {
                            return Err(Error::asdf_with_float(self.lexer.location()));
                        }
                        Err(e) => match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                self.builder.push_error(Error::integrity_check(
                                    self.lexer.location(),
                                    self.builder.decoded_len() - 1,
                                ));
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                self.phase = Phase::Data;
                self.state = State::Normal;

                Ok(())
            }
        }
    }

    fn compressed(&mut self) -> Result<()> {
        match self.phase {
            Phase::CheckPoint => {
                match self.parse_asdf() {
                    Ok(value) => {
                        self.builder.push_checkpoint_value(value as f64);
                    }
                    Err(e) => match e.kind() {
                        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                            self.builder.push_error(Error::overflow(self.lexer.location()));
                            self.builder.push_checkpoint_value(f64::NAN);
                        }
                        _ => unreachable!(),
                    }
                }
                self.phase = Phase::FirstData;

                Ok(())
            }
            Phase::Data | Phase::FirstData => {
                match self.state {
                    State::Normal | State::LastWasDifference(_) => match self.parse_asdf() {
                        Ok(value) => self.builder.push_decoded_i64(value),
                        Err(e) => match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                self.builder.push_error(Error::overflow_with_index(
                                    self.lexer.location(),
                                    self.builder.decoded_len(),
                                ));
                                self.builder.push_decoded_i64(i64::MIN);
                            }
                            _ => unreachable!(),
                        }
                    }
                    State::IntegrityCheck => match self.parse_asdf() {
                        Ok(value) => self.integrity_check(value)?,
                        Err(e) => match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                self.builder.push_error(Error::integrity_check(
                                    self.lexer.location(),
                                    self.builder.decoded_len() - 1,
                                ));
                            }
                            _ => unreachable!(),
                        }
                    }
                }

                self.phase = Phase::Data;
                self.state = State::Normal;

                Ok(())
            }
        }
    }

    fn difference(&mut self) -> Result<()> {
        if !self.builder.decoded_is_i64() {
            return Err(Error::asdf_with_float(self.lexer.location()));
        }

        let diff = self
            .parse_asdf()
            .map_err(|_| Error::dif_dup_overflow(self.lexer.location()))?;

        match self.phase {
            Phase::Data => {
                let value = self
                    .builder
                    .decoded_top()
                    .map(|top| *top + diff)
                    .ok_or_else(|| Error::asdf_after_check_point(self.lexer.location()))?;
                self.builder.push_decoded_i64(value);
                self.state = State::LastWasDifference(diff);

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => {
                Err(Error::asdf_after_check_point(self.lexer.location()))
            }
        }
    }

    fn duplicate(&mut self) -> Result<()> {
        if !self.builder.decoded_is_i64() {
            return Err(Error::asdf_with_float(self.lexer.location()));
        }

        let num = self
            .parse_asdf()
            .map_err(|_| Error::dif_dup_overflow(self.lexer.location()))?;

        match self.phase {
            Phase::Data => {
                let previous = self
                    .builder
                    .decoded_top()
                    .copied()
                    .ok_or_else(|| Error::asdf_after_check_point(self.lexer.location()))?;
                match self.state {
                    State::LastWasDifference(diff) => self
                        .builder
                        .extend_decoded((1..num).map(|i| previous + (diff * i))),
                    State::Normal => self
                        .builder
                        .extend_decoded(std::iter::repeat(previous).take((num - 1) as usize)),
                    _ => unreachable!(),
                }

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => {
                Err(Error::asdf_after_check_point(self.lexer.location()))
            }
        }
    }

    /// Handles [`Invalid`] tokens.
    ///
    /// Inserts [`i64::MIN`] and [`f64::NAN`] as sentinel values.
    ///
    /// [`Invalid`]: EncodedToken::Invalid
    fn invalid(&mut self) {
        match self.phase {
            Phase::Data | Phase::FirstData => {
                self.builder.push_error(Error::invalid_value_with_index(
                    self.lexer.location(),
                    self.builder.decoded_len()
                ));
                match self.state {
                    State::Normal | State::LastWasDifference(_) => {
                        self.builder.push_decoded_i64(i64::MIN);
                    }
                    State::IntegrityCheck => {
                        self.builder.push_error(Error::integrity_check(
                            self.lexer.location(),
                            self.builder.decoded_len() - 1,
                        ));
                    }
                }
                self.phase = Phase::Data;
                self.state = State::Normal;
            }
            Phase::CheckPoint => {
                self.builder.push_error(Error::invalid_value(self.lexer.location()));
                self.builder.push_checkpoint_value(f64::NAN);
                self.phase = Phase::FirstData;
            }
        }
    }

    /// Handles integrity checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoded values already contained `f64` values.
    ///
    /// # Panics
    ///
    /// Panics if the decoded stack is empty.
    fn integrity_check(&mut self, value: i64) -> Result<()> {
        if !self.builder.decoded_is_i64() {
            return Err(Error::asdf_with_float(self.lexer.location()));
        }

        let matches = self
            .builder
            .decoded_top()
            .map(|top| *top == value)
            .expect("integrity checks only after at least one value");
        if !matches {
            self.builder.push_error(Error::integrity_check(
                self.lexer.location(),
                self.builder.decoded_len() - 1,
            ));
            let top = self
                .builder
                .decoded_top_mut()
                .expect("integrity checks only after at least one value");
            *top = value;
        }

        Ok(())
    }

    /// Parse an `AFFN` numeric value.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing the value as an `i64` overflows.
    ///
    /// # Panics
    ///
    /// Panics if `lexer.slice()` cannot be parsed as an `i64` or `f64`.
    fn parse_affn(&self) -> std::result::Result<Affn, ParseIntError> {
        match self.lexer.slice().parse::<i64>() {
            Ok(int) => Ok(Affn::I64(int)),
            Err(e) => match e.kind() {
                IntErrorKind::InvalidDigit => {
                    let float = self
                        .lexer
                        .slice()
                        .parse::<f64>()
                        .expect("lexer (regex) should not match unparsable numerics");

                    // don't upgrade to float if it can be represented as i64
                    if float.fract() == 0.0 {
                        Ok(Affn::I64(float as i64))
                    } else {
                        Ok(Affn::F64(float))
                    }
                }
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => Err(e),
                _ => unreachable!(),
            }
        }
    }

    /// Parse an `ASDF` compressed, difference, or duplicate value.
    ///
    /// # Encoding
    ///
    /// An `ASDF` encoded value is composed of a leading character
    /// (@, %, A-Z, a-s), which determines the type of encoding, and a trailing
    /// numeric sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing the value as an `i64` overflows.
    ///
    /// # Panics
    ///
    /// Panics if `lexer.slice()` is not of the `ASDF` form.
    fn parse_asdf(&self) -> std::result::Result<i64, ParseIntError> {
        let encoded = self
            .lexer
            .slice()
            .chars()
            .next()
            .expect("lexer (regex) should have no empty ASDF tokens");
        let decoded = match encoded {
            '@' | '%' => 0,
            'A' | 'a' | 'J' | 'j' | 'S' => 1,
            'B' | 'b' | 'K' | 'k' | 'T' => 2,
            'C' | 'c' | 'L' | 'l' | 'U' => 3,
            'D' | 'd' | 'M' | 'm' | 'V' => 4,
            'E' | 'e' | 'N' | 'n' | 'W' => 5,
            'F' | 'f' | 'O' | 'o' | 'X' => 6,
            'G' | 'g' | 'P' | 'p' | 'Y' => 7,
            'H' | 'h' | 'Q' | 'q' | 'Z' => 8,
            'I' | 'i' | 'R' | 'r' | 's' => 9,
            _ => unreachable!("invalid ASDF character: {}", encoded),
        };
        let sign = match encoded {
            '@' | '%' | 'A'..='Z' | 's' => 1,
            'a'..='r' => -1,
            _ => unreachable!("invalid ASDF character: {}", encoded),
        };
        let numeric = &self.lexer.slice()[1..];
        let order = numeric.len() as u32;

        match numeric.parse::<i64>() {
            Ok(numeric) => Ok(sign * (decoded * 10_i64.pow(order) + numeric)),
            Err(e) => match e.kind() {
                IntErrorKind::Empty => Ok(sign * (decoded * 10_i64.pow(order))),
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => Err(e),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Position;
    use crate::jcampdx::{RawColumn, Table};
    use std::sync::LazyLock;

    static EXPECTED: LazyLock<DecodedBlock> = LazyLock::new(|| {
        let mut table = Table::new();
        table.push(RawColumn {
            id: "Unknown".to_string(),
            values: vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
        });

        DecodedBlock {
            table,
            errors: Vec::new(),
        }
    });

    macro_rules! decoder_test {
        ($name:ident, $data:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let decoded = Decoder::from(data).decode_source().unwrap();
                assert_eq!(decoded, *EXPECTED);
            }
        };
    }

    decoder_test!(
        affn,
        "\
            19        482       -763        215       -632\n\
            15       -924        357       -678        841\n\
            11        512       -194        321       -467\n\
            7        -689        278        505        732\n\
            3         835       -619        247       -193"
    );
    decoder_test!(
        pac,
        "\
            19 +482-763+215-632-924+357-678+841+512-194\n\
            9  +321-467-689+278+505+732+835-619+247-193"
    );
    decoder_test!(
        sqz,
        "\
            19 D82g63B15f32i24C57f78H41E12a94\n\
            9  C21d67f89B78E05G32H35f19B47a93"
    );
    decoder_test!(
        dif,
        "\
            19 D82j245R78q47k92J281j035J519l29p06\n\
            10 a94N15p88k22R67K27K27J03j454Q66m40\n\
            0  a93"
    );
    decoder_test!(
        difdup,
        "\
            19 D82j245R78q47k92J281j035J519l29p06\n\
            10 a94N15p88k22R67K27TJ03j454Q66m40\n\
            0  a93"
    );

    macro_rules! fatal_error_test {
        ($name:ident, $data:expr, $error:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let error = $error;
                let decoded = Decoder::from(data).decode_source().unwrap_err();
                assert_eq!(decoded, error);
            }
        };
    }

    fatal_error_test!(
        invalid_literal,
        "10 1 2 3 4 # 6 7 8 9 10",
        Error::invalid_literal(Position {
            line: 0,
            column: 11
        })
    );
    fatal_error_test!(
        dif_dup_check_point_value,
        "7 1 2 3 4\n\
         J 5 6 7 8",
        Error::asdf_after_check_point(Position { line: 1, column: 0 })
    );
    fatal_error_test!(
        dif_dup_value_after_check_point,
        "7 1 2 3 4\n\
         3 J 6 7 8",
        Error::asdf_after_check_point(Position { line: 1, column: 2 })
    );

    macro_rules! recoverable_error_test {
        ($name:ident, $data:expr, $errors:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let errors = $errors;
                let decoded = Decoder::from(data).decode_source().unwrap();
                assert_eq!(decoded.errors, errors);
            }
        };
    }

    recoverable_error_test!(
        overflow,
        "7 10000000000000000000 1 1 1\n\
         3 -10000000000000000000 1 1 1",
        [
            Error::overflow_with_index(Position { line: 0, column: 2 }, 0),
            Error::overflow_with_index(Position { line: 1, column: 2 }, 4),
        ]
    );
    recoverable_error_test!(
        integrity_check,
        "7 A0LnK\n\
         4 IK%nM\n\
         0 G",
        [
            Error::integrity_check(Position { line: 1, column: 2 }, 3),
            Error::integrity_check(Position { line: 2, column: 2 }, 7),
        ]
    );
    recoverable_error_test!(
        invalid_value,
        "7 1 1 ? 1\n\
         3 0 ? 1 ?",
        [
            Error::invalid_value_with_index(Position { line: 0, column: 6 }, 2),
            Error::invalid_value_with_index(Position { line: 1, column: 4 }, 5),
            Error::invalid_value_with_index(Position { line: 1, column: 8 }, 7),
        ]
    );
}
