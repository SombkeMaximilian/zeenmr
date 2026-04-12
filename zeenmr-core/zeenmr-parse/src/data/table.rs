use crate::data::{Column, Value};
use std::borrow::{Borrow, Cow};
use std::collections::BTreeMap;

/// Type alias for tables containing raw data.
pub type DataTable<'source> = Table<'source, Cow<'source, str>, Column<'source>>;

/// Type alias for tables containing dataset parameters.
pub type ParameterTable<'source> = Table<'source, Cow<'source, str>, Value<'source>>;

/// Table in a JCAMP-DX dataset.
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub struct Table<'source, K, V> {
    /// Identifier of the table, if any.
    id: Option<Cow<'source, str>>,
    /// Entries in the table.
    map: BTreeMap<K, V>,
}

impl<'source, K, V> FromIterator<(K, V)> for Table<'source, K, V>
where
    K: Ord,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            id: None,
            map: BTreeMap::from_iter(iter),
        }
    }
}

impl<'source, K, V> Table<'source, K, V> {
    /// Constructs a new, empty `Table`.
    pub fn new() -> Self {
        Self {
            id: None,
            map: BTreeMap::new(),
        }
    }

    /// Constructs a new, empty `Table` with an identifier.
    pub fn new_with_id<T: Into<Cow<'source, str>>>(id: T) -> Self {
        Self {
            id: Some(id.into()),
            map: BTreeMap::new(),
        }
    }

    /// Returns the identifier of the table as a string slice, if any.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Sets the identifier of the table.
    pub fn set_id<T: Into<Cow<'source, str>>>(&mut self, id: T) {
        self.id = Some(id.into());
    }

    /// Returns the number of entries in the table.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the table contains no elements.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.contains_key(key)
    }

    /// Returns a reference to the corresponding value of the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.get(key)
    }

    /// Returns references to the key-value pair matching the given key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.get_key_value(key)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.get_mut(key)
    }

    /// Inserts a key-value pair into the table.
    ///
    /// Returns `None` if the table did not have this key present. Otherwise,
    /// updates the value and returns the old value.
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        self.map.insert(key, value)
    }

    /// Removes all elements from the table.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Removes a key from the table, returning the corresponding value if the
    /// key was previously in the table.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.remove(key)
    }

    /// Removes a key from the table, returning the key-value pair if the key
    /// was previously in the table.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.map.remove_entry(key)
    }

    /// Returns an iterator over the key-value pairs of the table.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }

    /// Returns a mutable iterator over the key-value pairs of the table.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.map.iter_mut()
    }

    /// Returns an iterator over the keys of the table.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    /// Returns an iterator over the values of the table.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }

    /// Returns a mutable iterator over the values of the table.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.map.values_mut()
    }
}

impl<'source> ParameterTable<'source> {
    /// Converts this `Table` into an owned form with a `'static` lifetime.
    ///
    /// This is useful when you need to store a parsed `Table` beyond the
    /// lifetime of the input buffer. Borrowed string data is cloned into
    /// `Cow::Owned`. Array types are converted recursively. Everything else is
    /// moved.
    pub fn into_owned(self) -> ParameterTable<'static> {
        let id = self.id.map(|id| Cow::Owned(id.into_owned()));
        let map = self
            .map
            .into_iter()
            .map(|(key, value)| (Cow::Owned(key.into_owned()), value.into_owned()))
            .collect();

        Table { id, map }
    }
}

impl<'source> DataTable<'source> {
    /// Converts this `Table` into an owned form with a `'static` lifetime.
    ///
    /// This is useful when you need to store a parsed `Table` beyond the
    /// lifetime of the input buffer. Borrowed string data is cloned into
    /// `Cow::Owned`. Array types are converted recursively. Everything else is
    /// moved.
    pub fn into_owned(self) -> DataTable<'static> {
        let id = self.id.map(|id| Cow::Owned(id.into_owned()));
        let map = self
            .map
            .into_iter()
            .map(|(key, value)| (Cow::Owned(key.into_owned()), value.into_owned()))
            .collect();

        Table { id, map }
    }
}
