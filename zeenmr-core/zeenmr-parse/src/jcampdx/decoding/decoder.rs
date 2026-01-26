use crate::Location;
use crate::jcampdx::decoding::error::{Error, Kind, Result};
use crate::jcampdx::decoding::{DecodedBlock, DecodedBlockBuilder, EncodedToken};
use crate::location::Cursor;
use logos::{Lexer, Logos};

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
        while let Some(token) = self.lexer.next() {
            match token {
                Ok(EncodedToken::CheckPoint) => self.check_point(),
                Ok(EncodedToken::Numeric(value)) => self.numeric(value),
                Ok(EncodedToken::Compressed(value)) => self.numeric(value),
                Ok(EncodedToken::Difference(diff)) => self.difference(diff)?,
                Ok(EncodedToken::Duplicate(num)) => self.duplicate(num as usize)?,
                Ok(EncodedToken::Invalid(position)) => {
                    self.builder
                        .push_error(Error::invalid_value(position, self.builder.decoded_len()));
                    self.numeric(i64::MIN);
                }
                Ok(EncodedToken::End) => break,
                Err(e) => match e.kind() {
                    Kind::Overflow => self.overflow(),
                    Kind::InvalidLiteral | Kind::UnsupportedFormat => return Err(e),
                    _ => unreachable!(),
                },
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

    fn numeric(&mut self, value: i64) {
        match self.phase {
            Phase::Data | Phase::FirstData => {
                match self.state {
                    State::Normal | State::LastWasDifference(_) => {
                        self.builder.push_decoded_value(value);
                    }
                    State::IntegrityCheck => {
                        let integrity_check_result = self
                            .builder
                            .decoded_top()
                            .map(|top| *top == value)
                            .expect("integrity checks only after at least one value");

                        if !integrity_check_result {
                            self.builder.push_error(Error::integrity_check(
                                self.lexer.location(),
                                self.builder.decoded_len() - 1,
                            ));
                            *(self.builder.decoded_top_mut().unwrap()) = value;
                        }
                    }
                }
                self.phase = Phase::Data;
                self.state = State::Normal;
            }
            Phase::CheckPoint => {
                self.builder.push_checkpoint_value(value as f64);
                self.phase = Phase::FirstData;
            }
        }
    }

    fn difference(&mut self, diff: i64) -> Result<()> {
        match self.phase {
            Phase::Data => {
                let value = self
                    .builder
                    .decoded_top()
                    .map(|top| *top + diff)
                    .ok_or_else(|| Error::dif_dup_after_check_point(self.lexer.location()))?;
                self.builder.push_decoded_value(value);
                self.state = State::LastWasDifference(diff);

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => {
                Err(Error::dif_dup_after_check_point(self.lexer.location()))
            }
        }
    }

    fn duplicate(&mut self, num: usize) -> Result<()> {
        match self.phase {
            Phase::Data => {
                let previous = self
                    .builder
                    .decoded_top()
                    .copied()
                    .ok_or_else(|| Error::dif_dup_after_check_point(self.lexer.location()))?;
                match self.state {
                    State::LastWasDifference(diff) => self
                        .builder
                        .extend_decoded((1..num as i64).map(|i| previous + (diff * i))),
                    State::Normal => self
                        .builder
                        .extend_decoded(std::iter::repeat(previous).take(num - 1)),
                    _ => unreachable!(),
                }

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => {
                Err(Error::dif_dup_after_check_point(self.lexer.location()))
            }
        }
    }

    fn overflow(&mut self) {
        match self.phase {
            Phase::Data | Phase::FirstData => {
                self.builder
                    .push_error(Error::overflow_with_index(
                        self.lexer.location(),
                        self.builder.decoded_len(),
                    ));
                self.numeric(i64::MIN);
            }
            Phase::CheckPoint => {
                self.builder
                    .push_error(Error::overflow(self.lexer.location()));
                self.numeric(i64::MIN);
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
        Error::dif_dup_after_check_point(Position { line: 1, column: 0 })
    );
    fatal_error_test!(
        dif_dup_value_after_check_point,
        "7 1 2 3 4\n\
         3 J 6 7 8",
        Error::dif_dup_after_check_point(Position { line: 1, column: 2 })
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
            Error::invalid_value(Position { line: 0, column: 6 }, 2),
            Error::invalid_value(Position { line: 1, column: 4 }, 5),
            Error::invalid_value(Position { line: 1, column: 8 }, 7),
        ]
    );
}
