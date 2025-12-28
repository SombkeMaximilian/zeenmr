use crate::Location;
use crate::jcampdx::decoding::error::{Error, Kind, Result};
use crate::jcampdx::decoding::{CheckPoint, DecodedBlock, DecodeExit, EncodedToken};
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
    IntegrityCheck(i64),
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
    /// Decoded intensity values.
    decoded: Vec<i64>,
    /// Checkpoint indices in the sequence.
    check_points: Vec<usize>,
    /// Checkpoint values.
    check_point_values: Vec<i64>,
    /// Non-fatal errors that occur decoding.
    errors: Vec<Error>,
}

impl<'source> From<&'source str> for Decoder<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: EncodedToken::lexer(value),
            phase: Phase::CheckPoint,
            state: State::Normal,
            decoded: Vec::new(),
            check_points: vec![0],
            check_point_values: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl<'source> Decoder<'source> {
    pub(crate) fn decode_source(mut self) -> Result<DecodeExit<'source, i64>> {
        while let Some(token) = self.lexer.next() {
            match token {
                Ok(EncodedToken::CheckPoint) => self.check_point(),
                Ok(EncodedToken::Numeric(value)) => self.numeric(value),
                Ok(EncodedToken::Compressed(value)) => self.numeric(value),
                Ok(EncodedToken::Difference(diff)) => self.difference(diff)?,
                Ok(EncodedToken::Duplicate(num)) => self.duplicate(num as usize)?,
                Ok(EncodedToken::Invalid(position)) => {
                    self.errors.push(Error::invalid_value(position, self.decoded.len()));
                    self.numeric(i64::MIN);
                },
                Ok(EncodedToken::End) => {
                    let lexer = self.lexer.clone();

                    return Ok(DecodeExit::HeaderKey(self.finalize(), lexer));
                },
                Err(e) => match e.kind() {
                    Kind::InvalidLiteral => return Err(e),
                    _ => self.errors.push(e),
                },
            }
        }

        Ok(DecodeExit::EndOfInput(self.finalize()))
    }

    fn finalize(self) -> DecodedBlock<i64> {
        DecodedBlock::new(
            self.decoded,
            self.check_points
                .into_iter()
                .zip(self.check_point_values.into_iter())
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            self.errors,
        )
    }

    fn check_point(&mut self) {
        match self.state {
            State::LastWasDifference(_) => {
                self.state = State::IntegrityCheck(*(self.decoded.last().unwrap()));
                self.check_points.push(self.decoded.len() - 1);
            }
            _ => {
                self.check_points.push(self.decoded.len());
            }
        }
        self.phase = Phase::CheckPoint;
    }

    fn numeric(&mut self, value: i64) {
        match self.phase {
            Phase::Data | Phase::FirstData => {
                match self.state {
                    State::Normal | State::LastWasDifference(_) => {
                        self.decoded.push(value);
                    }
                    State::IntegrityCheck(check) => {
                        if value != check {
                            self.errors.push(Error::integrity_check(
                                self.lexer.location(),
                                self.decoded.len() - 1,
                            ));
                            *(self.decoded.last_mut().unwrap()) = value;
                        }
                    }
                }
                self.phase = Phase::Data;
                self.state = State::Normal;
            }
            Phase::CheckPoint => {
                self.check_point_values.push(value);
                self.phase = Phase::FirstData;
            }
        }
    }

    fn difference(&mut self, diff: i64) -> Result<()> {
        match self.phase {
            Phase::Data => {
                let result = *(self.decoded.last().unwrap()) + diff;
                self.decoded.push(result);
                self.state = State::LastWasDifference(diff);

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => Err(Error::dif_dup_after_check_point(self.lexer.location())),
        }
    }

    fn duplicate(&mut self, num: usize) -> Result<()> {
        match self.phase {
            Phase::Data => {
                let previous = *(self.decoded.last().unwrap());
                match self.state {
                    State::LastWasDifference(diff) => {
                        let values = (1..num).map(|i| previous + (diff * i as i64));
                        self.decoded.extend(values);
                    }
                    State::Normal => {
                        self.decoded
                            .extend(std::iter::repeat(previous).take(num));
                    }
                    _ => {}
                }

                Ok(())
            }
            Phase::CheckPoint | Phase::FirstData => Err(Error::dif_dup_after_check_point(self.lexer.location())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Position;

    macro_rules! decoder_test {
        ($name:ident, $data:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $expected;
                let decoded = Decoder::from(data).decode_source().unwrap();
                assert_eq!(*(decoded.block()), expected);
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
            3         835       -619        247       -193",
        DecodedBlock::new(
            vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
            [(0, 19), (4, 15), (8, 11), (12, 7), (16, 3)]
                .into_iter()
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            Vec::new(),
        )
    );
    decoder_test!(
        pac,
        "\
            19 +482-763+215-632-924+357-678+841+512-194\n\
            9  +321-467-689+278+505+732+835-619+247-193",
        DecodedBlock::new(
            vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
            [(0, 19), (10, 9)]
                .into_iter()
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            Vec::new(),
        )
    );
    decoder_test!(
        sqz,
        "\
            19 D82g63B15f32i24C57f78H41E12a94\n\
            9  C21d67f89B78E05G32H35f19B47a93",
        DecodedBlock::new(
            vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
            [(0, 19), (10, 9)]
                .into_iter()
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            Vec::new(),
        )
    );
    decoder_test!(
        dif,
        "\
            19 D82j245R78q47k92J281j035J519l29p06\n\
            10 a94N15p88k22R67K27K27J03j454Q66m40\n\
            0  a93",
        DecodedBlock::new(
            vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
            [(0, 19), (9, 10), (19, 0)]
                .into_iter()
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            Vec::new(),
        )
    );
    decoder_test!(
        difdup,
        "\
            19 D82j245R78q47k92J281j035J519l29p06\n\
            10 a94N15p88k22R67K27TJ03j454Q66m40\n\
            0  a93",
        DecodedBlock::new(
            vec![
                482, -763, 215, -632, -924, 357, -678, 841, 512, -194, 321, -467, -689, 278, 505,
                732, 835, -619, 247, -193,
            ],
            [(0, 19), (9, 10), (19, 0)]
                .into_iter()
                .map(|(index, value)| CheckPoint::new(index, value))
                .collect(),
            Vec::new(),
        )
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
        Error::invalid_literal(Position { line: 0, column: 11 })
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
                assert_eq!(decoded.block().errors(), errors);
            }
        };
    }
}
