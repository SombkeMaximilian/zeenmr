use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of a deconvolution.
///
/// A `Deconvolution` contains the deconvoluted signals as peak shapes, the
/// settings used for deconvolution, and the mean squared error of between the
/// original spectrum and the superposition of peak shapes.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Deconvolution` implements the
/// [`Serialize`] and [`Deserialize`] traits provided that the contained
/// types also implement them. This is guaranteed for any types provided by
/// this crate.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, P: Serialize",
        deserialize = "T: Deserialize<'de>, P: Deserialize<'de>"
    ))
)]
pub struct Deconvolution<T, P> {
    /// Mean squared error of the deconvolution.
    mse: T,
    /// Deconvoluted peak shapes.
    #[cfg_attr(feature = "serde", serde(with = "serialize_peak_shapes"))]
    peak_shapes: Arc<[P]>,
}

// Enabling the `rc` feature of serde would force it on everyone using this
// crate, so we just implement this manually for now. If `Arc` turns out to be
// unnecessary, we can switch to `Box<[P]>` later and remove this module.
#[cfg(feature = "serde")]
mod serialize_peak_shapes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub(crate) fn serialize<P, S>(peak_shapes: &Arc<[P]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        P: Serialize,
        S: Serializer,
    {
        peak_shapes.as_ref().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, P, D>(deserializer: D) -> Result<Arc<[P]>, D::Error>
    where
        P: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Box::<[P]>::deserialize(deserializer)?.into())
    }
}

impl<T, P> Deconvolution<T, P> {
    /// Creates a new `Deconvolution`.
    ///
    /// Normally, this type is only instantiated by the deconvolution functions
    /// of deconvoluters.
    pub fn new<I>(peak_shapes: I, mse: T) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self {
            peak_shapes: Arc::from_iter(peak_shapes),
            mse,
        }
    }

    /// Returns the deconvoluted peak shapes.
    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }

    /// Returns the mean squared error of the deconvolution.
    pub fn mse(&self) -> &T {
        &self.mse
    }
}
