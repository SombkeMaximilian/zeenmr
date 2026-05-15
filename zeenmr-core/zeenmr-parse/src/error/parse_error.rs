use crate::error::{ByteRange, RangeLabel};
use std::borrow::Cow;
use std::ops::ControlFlow;

/// Specialized error trait for parser errors.
///
/// The [`Error`] type uses its methods to render error messages.
pub trait ParseError: std::error::Error {
    /// Short, one-line description of what went wrong.
    fn message(&self) -> Cow<'static, str>;

    /// Labeled byte ranges that are relevant for the error.
    ///
    /// The iterator must always return at least one element, and the byte range
    /// within each element must not span across multiple lines.
    fn labels(&self, source: &str) -> Vec<RangeLabel>;

    /// Optional, potentially longer explanations of the error, if any.
    fn notes(&self, source: &str) -> Vec<Cow<'static, str>>;

    /// Optional hints for how to fix the error, if any.
    fn fix_hints(&self, source: &str) -> Vec<Cow<'static, str>>;
}

/// Trait for creating the pretty-print [`Error`].
pub trait AttachSource: Sized {
    /// Attaches the source string which caused the parser error.
    fn with_source(self, source: &str) -> Error<'_, Self>;
}

impl<E> AttachSource for E
where
    E: ParseError + Sized,
{
    fn with_source(self, source: &str) -> Error<'_, Self> {
        Error {
            error: self,
            source,
            filename: None,
        }
    }
}

/// Tracks the occupied byte ranges on each output row.
#[derive(Debug)]
struct ColumnOccupancy(Vec<Vec<ByteRange>>);

impl ColumnOccupancy {
    /// Returns the index of the first row where `start..end` is free,
    /// allocating a new row if needed.
    fn claim_row(&mut self, start: usize, end: usize) -> usize {
        let column_range = ByteRange::from(start..end);
        let row = self
            .0
            .iter()
            .position(|occupied| {
                !occupied
                    .iter()
                    .any(|range| range.overlaps(&column_range))
            })
            .unwrap_or_else(|| {
                self.0.push(Vec::new());

                self.0.len() - 1
            });
        self.0[row].push(column_range);

        row
    }
}

/// Group of labels on the same line.
#[derive(Clone, Eq, PartialEq, Debug)]
struct LabelGroup<'source> {
    /// Line start byte offsets.
    offset: usize,
    /// 1-based line number.
    line: usize,
    /// Content of the line.
    text: &'source str,
    /// Labels to render below the content.
    labels: Vec<RangeLabel>,
}

impl<'source> LabelGroup<'source> {
    const TAG: &'static str = r"\_ ";

    /// Returns the corresponding markers for primary and secondary labels.
    fn marker(is_cause: bool) -> char {
        if is_cause { '^' } else { '-' }
    }

    /// Renders the label group.
    ///
    /// The provided gutter must not be empty, otherwise the line will not be
    /// properly displayed.
    fn render(&self, f: &mut std::fmt::Formatter, gutter: &str) -> std::fmt::Result {
        let line_offset = self.offset;
        let column = move |offset: usize| -> usize { offset.saturating_sub(line_offset) };

        let mut occupancy = ColumnOccupancy(Vec::new());
        let highlight_rows = self
            .labels
            .iter()
            .map(|label| {
                let start = column(label.range.start);
                let end = column(label.range.end).max(start + 1);
                let marker = Self::marker(label.is_cause);
                let highlight = std::iter::repeat_n(marker, end - start).collect::<String>();

                (occupancy.claim_row(start, end), start, highlight)
            })
            .collect::<Vec<(usize, usize, String)>>();
        let label_rows = self
            .labels
            .iter()
            .filter_map(|label| {
                label.label.as_deref().map(|label_text| {
                    let start = column(label.range.start);
                    let end = (start + label_text.len() + Self::TAG.len()).max(start + 1);
                    let tagged_label = format!("{}{}", Self::TAG, label_text);

                    (occupancy.claim_row(start, end), start, tagged_label)
                })
            })
            .collect::<Vec<(usize, usize, String)>>();

        let mut rows = vec![String::new(); occupancy.0.len()];
        let mut write_commands = highlight_rows
            .into_iter()
            .chain(label_rows)
            .collect::<Vec<(usize, usize, String)>>();
        write_commands.sort_unstable(); // sort by row, then column
        for (row, column, write) in write_commands.into_iter() {
            let write_row = &mut rows[row];
            write_row.extend(std::iter::repeat_n(' ', column - write_row.len()));
            write_row.push_str(&write);
        }

        writeln!(
            f,
            "{:>width$} | {}",
            self.line,
            self.text,
            width = gutter.len()
        )?;
        for row in rows {
            writeln!(f, "{gutter} | {row}")?;
        }

        Ok(())
    }
}

/// General parser error with a reference to the parsed source.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Error<'source, E> {
    /// Error returned from a parser.
    error: E,
    /// Source string that was attempted to be parsed.
    source: &'source str,
    /// Optional filename.
    filename: Option<Cow<'static, str>>,
}

impl<'source, E> std::fmt::Display for Error<'source, E>
where
    E: ParseError,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "error: {}", self.error.message())?;

        let label_groups = self.label_groups();
        let (cause_line, cause_column) = label_groups
            .iter()
            .find_map(|group| {
                group
                    .labels
                    .iter()
                    .find(|label| label.is_cause)
                    .map(|label| {
                        (
                            group.line,
                            label.range.start.saturating_sub(group.offset) + 1,
                        )
                    })
            })
            .map_or_else(
                || {
                    writeln!(
                        f,
                        "WARNING: no error cause provided, please file a bug report"
                    )
                    .map(|_| (0, 0))
                },
                Ok,
            )?;
        let file = self.filename.as_deref().unwrap_or("<input>");
        let gutter_width = 1 + label_groups
            .iter()
            .map(|group| group.line)
            .max()
            .unwrap_or(1)
            .ilog10() as usize;
        let gutter = " ".repeat(gutter_width);

        writeln!(f, "{gutter} ==> {file}:{cause_line}:{cause_column}")?;
        writeln!(f, "{gutter} |")?;
        for group in label_groups.into_iter() {
            group.render(f, &gutter)?;
            writeln!(f, "{gutter} |")?;
        }
        for note in self.error.notes(self.source).into_iter() {
            writeln!(f, "{gutter} == note: {note}")?;
        }
        for hint in self.error.fix_hints(self.source).into_iter() {
            writeln!(f, "{gutter} == hint: {hint}")?;
        }

        Ok(())
    }
}

impl<'source, E> Error<'source, E> {
    /// Attaches a filename for display purposes.
    pub fn with_filename(mut self, filename: String) -> Self {
        self.filename = Some(filename.into());

        self
    }

    /// Returns a reference to the contained error.
    pub fn error(&self) -> &E {
        &self.error
    }

    /// Returns the source string which caused the parser error.
    pub fn source_str(&self) -> &'source str {
        self.source
    }

    /// Returns a reference to the filename, if any.
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }
}

impl<'source, E> Error<'source, E>
where
    E: ParseError,
{
    /// Groups the labels returned by the error by their lines.
    fn label_groups(&self) -> Vec<LabelGroup<'_>> {
        let mut labels = self.error.labels(self.source);
        labels.sort_unstable();

        let mut label_groups = Vec::<LabelGroup>::new();
        let mut source_bytes = self.source.bytes().enumerate();
        let mut line = 1_usize;
        let mut start_offset = 0_usize;
        while !labels.is_empty() {
            let flow = loop {
                match source_bytes.next() {
                    Some((i, b'\n')) => break ControlFlow::Continue(i),
                    Some((i, b'\r')) => match source_bytes.next() {
                        Some((_, b'\n')) => break ControlFlow::Continue(i + 1),
                        _ => break ControlFlow::Continue(i),
                    },
                    Some(_) => continue,
                    None => break ControlFlow::Break(self.source.len()),
                }
            };
            let end_offset = match flow {
                ControlFlow::Continue(i) | ControlFlow::Break(i) => i,
            };
            let line_range = start_offset..end_offset;
            let labels_on_line = labels
                .iter()
                .position(|label| !(label.range.end <= end_offset))
                .unwrap_or(labels.len());
            if labels_on_line > 0 {
                label_groups.push(LabelGroup {
                    offset: start_offset,
                    line,
                    text: &self.source[line_range],
                    labels: labels.drain(..labels_on_line).collect(),
                })
            }
            line += 1;
            start_offset = end_offset + 1;
            if flow.is_break() {
                break;
            }
        }

        label_groups
    }
}
