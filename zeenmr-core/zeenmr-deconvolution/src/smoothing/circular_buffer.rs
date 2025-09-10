use std::collections::VecDeque;

/// FIFO buffer with a fixed capacity that wraps around and overwrites old
/// elements when full.
///
/// # Optimization
///
/// While [`VecDeque`] is not the most performant data structure for this use
/// case, the smoothing part of the deconvolution pipeline takes up a small
/// fraction of the total runtime. Any further optimization would likely not
/// yield any significant improvements.
#[derive(Clone, Debug)]
pub(crate) struct CircularBuffer<T> {
    /// Wrapper around a double-ended queue to store the elements.
    buffer: VecDeque<T>,
    /// Maximum number of elements the buffer can hold.
    ///
    /// This is stored separately because [`VecDeque`] only guarantees that it
    /// can hold at least this many elements.
    capacity: usize,
}

impl<T> CircularBuffer<T> {
    /// Creates a new `CircularBuffer` with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the capacity is zero.
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be greater than zero");

        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Pushes a new element into the buffer and returns the oldest element if
    /// the buffer was already full, or `None` otherwise.
    pub(crate) fn push(&mut self, value: T) -> Option<T> {
        let popped = if self.buffer.len() == self.capacity {
            self.pop()
        } else {
            None
        };
        self.buffer.push_back(value);

        popped
    }

    /// Pops and returns the oldest element from the buffer, or `None` if the
    /// buffer was already empty.
    pub(crate) fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    /// Clears the buffer, removing all elements.
    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Returns the number of elements currently in the buffer.
    pub(crate) fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push() {
        let mut buffer = CircularBuffer::<u32>::new(2);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.push(3), Some(1));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn pop() {
        let mut buffer = CircularBuffer::<u32>::new(2);
        buffer.push(1);
        buffer.push(2);
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), None);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn clear() {
        let mut buffer = CircularBuffer::<u32>::new(2);
        buffer.push(1);
        buffer.push(2);
        buffer.clear();
        assert_eq!(buffer.pop(), None);
        assert_eq!(buffer.len(), 0);
    }
}
