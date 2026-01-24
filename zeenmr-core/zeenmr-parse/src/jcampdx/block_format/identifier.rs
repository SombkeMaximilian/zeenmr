#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct Identifier {
    name: String,
    offset: Option<usize>,
}

impl From<&str> for Identifier {
    fn from(value: &str) -> Self {
        Self {
            name: value.into(),
            offset: None,
        }
    }
}

impl From<Identifier> for String {
    fn from(value: Identifier) -> Self {
        value.name
    }
}

impl Identifier {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn offset(&self) -> Option<usize> {
        self.offset
    }
}
