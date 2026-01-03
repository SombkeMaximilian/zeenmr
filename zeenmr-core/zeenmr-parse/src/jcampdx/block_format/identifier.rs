#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct Identifier {
    name: Box<str>,
    offset: Option<usize>,
}

impl From<String> for Identifier {
    fn from(value: String) -> Self {
        Self {
            name: value.into(),
            offset: None,
        }
    }
}

impl From<Box<str>> for Identifier {
    fn from(value: Box<str>) -> Self {
        Self {
            name: value,
            offset: None,
        }
    }
}

impl From<&str> for Identifier {
    fn from(value: &str) -> Self {
        Self {
            name: value.into(),
            offset: None,
        }
    }
}

impl Identifier {
    pub(crate) fn new<T>(name: T, offset: usize) -> Self
    where
        T: AsRef<str>,
    {
        Self {
            name: name.as_ref().into(),
            offset: Some(offset),
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn offset(&self) -> Option<usize> {
        self.offset
    }
}
