use crate::error::ByteRange;

/// Main trait for structured parser errors.
pub trait ParseError: std::error::Error {
    /// Primary position that triggered the error.
    fn primary(&self) -> ByteRange;

    /// Short, one-line description of what went wrong.
    fn message(&self) -> String;

    /// Text next to the highlight.
    fn highlight_text(&self) -> String;

    /// Optional, longer explanations of the error.
    fn note(&self) -> Annotations {
        Annotations::default()
    }

    /// Optional hints for how to fix the error.
    fn fix_hint(&self) -> Annotations {
        Annotations::default()
    }

    /// Returns the cause of the error, if any.
    fn cause(&self) -> Option<&(dyn ParseError + Send + Sync)> {
        None
    }

    /// Secondary positions with short descriptions involved in the error.
    ///
    /// Requires the source to make errors as slim as possible.
    #[allow(unused_variables)]
    fn secondary(&self, source: &str) -> Vec<(ByteRange, String)> {
        Vec::new()
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub enum Annotations {
    #[default]
    None,
    One(String),
    Multiple(Vec<String>),
}

impl From<&'static str> for Annotations {
    fn from(value: &'static str) -> Self {
        Annotations::One(value.into())
    }
}

impl From<String> for Annotations {
    fn from(value: String) -> Self {
        Annotations::One(value)
    }
}

impl From<Vec<String>> for Annotations {
    fn from(value: Vec<String>) -> Self {
        Annotations::Multiple(value)
    }
}

/// Error type for displaying information about the error.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct ErrorDisplay<'source, E> {
    /// Error that occurred.
    error: E,
    /// Input source.
    source: &'source str,
    /// File name of the source, if any.
    filename: Option<&'source str>,
}

impl<'source, E> std::error::Error for ErrorDisplay<'source, E> where E: ParseError {}

impl<'source, E> std::fmt::Display for ErrorDisplay<'source, E>
where
    E: ParseError,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "error: {}", self.error.message())?;

        let position = self.error.primary();
        let start = self.source[..position.start]
            .rfind(['\n', '\r'])
            .map(|i| {
                if self.source.as_bytes().get(i) == Some(&b'\r')
                    && self.source.as_bytes().get(i + 1) == Some(&b'\n')
                {
                    i + 2
                } else {
                    i + 1
                }
            })
            .unwrap_or(0);
        let end = self.source[start..]
            .find(['\n', '\r'])
            .map(|i| i + start)
            .unwrap_or(self.source.len());
        let source_line = &self.source[start..end];

        let column = self.source[start..position.start].chars().count();
        let file = self.filename.unwrap_or("<input>");
        let line = format!("{}", todo!());
        let gutter = " ".repeat(line.len());

        writeln!(f, "{gutter} ---> {file}:{line}:{}", column + 1)?;
        writeln!(f, "{gutter} |")?;
        writeln!(f, "{line} | {source_line}")?;

        let highlight = "^".repeat((position.end - position.start).max(1));
        let pad = " ".repeat(column);
        writeln!(
            f,
            "{gutter} | {pad}{highlight} {}",
            self.error.highlight_text()
        )?;
        writeln!(f, "{gutter} |",)?;

        match self.error.note() {
            Annotations::None => {}
            Annotations::One(note) => {
                writeln!(f, "{gutter} - note: {note}")?;
            }
            Annotations::Multiple(notes) => {
                for note in notes.into_iter() {
                    writeln!(f, "{gutter} - note: {note}")?;
                }
            }
        }

        match self.error.fix_hint() {
            Annotations::None => {}
            Annotations::One(hint) => {
                writeln!(f, "{gutter} - hint: {hint}")?;
            }
            Annotations::Multiple(hints) => {
                for hint in hints.into_iter() {
                    writeln!(f, "{gutter} - hint: {hint}")?;
                }
            }
        }

        Ok(())
    }
}

impl<'source, E> ErrorDisplay<'source, E> {
    /// Constructs a new `ErrorDisplay`.
    pub fn new(error: E, source: &'source str) -> Self {
        Self {
            error,
            source,
            filename: None,
        }
    }

    /// Adds a filename to the `ErrorDisplay`.
    pub fn with_filename(mut self, filename: &'source str) -> Self {
        self.filename = Some(filename);

        self
    }
}
