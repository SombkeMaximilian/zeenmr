use std::ops::{Deref, DerefMut};

/// Maximum number of static dimensions.
///
/// Up to 3D NMR can be statically supported.
const MAX_STATIC_DIMS: usize = 3;

/// Array dimension index.
///
/// An array generally has `N` dimensions. This type encapsulates the index `i`
/// of such a dimension with `0 <= i < N`.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct DimIndex(usize);

impl DimIndex {
    /// Returns the index of the dimension.
    #[inline(always)]
    pub fn index(&self) -> usize {
        // perhaps the contained `usize` could just be pub
        self.0
    }
}

pub trait Dimensions: Clone + Eq + PartialEq + AsRef<[usize]> + AsMut<[usize]> {
    /// Compile time constant rank, if available.
    const RANK: Option<usize>;

    /// Returns a value of the given rank with all entries zero, or `None` if
    /// this type cannot represent it.
    fn zeroed(rank: usize) -> Option<Self>;

    /// Returns the rank of `self`.
    fn rank(&self) -> usize {
        self.as_ref().len()
    }
}

impl<const N: usize> Dimensions for [usize; N] {
    const RANK: Option<usize> = Some(N);

    fn zeroed(rank: usize) -> Option<Self> {
        if rank == N {
            Some([0; N])
        } else {
            None
        }
    }

    fn rank(&self) -> usize {
        N
    }
}

#[derive(Clone, Debug)]
enum DynDimInner {
    Stack(StaticLen, [usize; MAX_STATIC_DIMS]),
    Heap(Vec<usize>),
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
#[repr(usize)]
enum StaticLen {
    #[default]
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}

impl StaticLen {
    /// Returns the equivalent `StaticLen`, or `None` if it exceeds
    /// `MAX_STATIC_DIMS`.
    fn from_usize(n: usize) -> Option<Self> {
        match n {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            _ => None,
        }
    }
}

impl Deref for DynDimInner {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Stack(n, dims) => &dims[..*n as usize],
            Self::Heap(dims) => dims,
        }
    }
}

impl DerefMut for DynDimInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Stack(n, dims) => &mut dims[..*n as usize],
            Self::Heap(dims) => dims,
        }
    }
}

impl Default for DynDimInner {
    fn default() -> Self {
        Self::Stack(StaticLen::Zero, [0; MAX_STATIC_DIMS])
    }
}

impl DynDimInner {
    fn from_array<const N: usize>(value: [usize; N]) -> Self {
        if N <= MAX_STATIC_DIMS {
            let len = StaticLen::from_usize(N)
                .expect("above `if` should make this infallible");
            let mut dims = [0; MAX_STATIC_DIMS];
            dims[..value.len()].copy_from_slice(&value);

            Self::Stack(len, dims)
        } else {
            Self::Heap(value.to_vec())
        }
    }

    fn from_vec(value: Vec<usize>) -> Self {
        if value.len() <= MAX_STATIC_DIMS {
            let len = StaticLen::from_usize(value.len())
                .expect("above `if` should make this infallible");
            let mut dims = [0; MAX_STATIC_DIMS];
            dims[..value.len()].copy_from_slice(&value);

            Self::Stack(len, dims)
        } else {
            Self::Heap(value)
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynDim(DynDimInner);
