use num_traits::Zero;
use std::collections::VecDeque;

/// FIFO buffer with a fixed capacity that wraps around and overwrites old
/// elements when full.
#[derive(Debug)]
pub(crate) struct CircularBuffer<T> {
    /// The underlying storage for the buffer.
    buffer: VecDeque<T>,
    /// The maximum number of elements the buffer can hold.
    ///
    /// This has to be stored separately because [`VecDeque`] only guarantees
    /// that it can hold at least this many elements.
    capacity: usize,
}

impl<T: Copy + Zero> CircularBuffer<T> {
    /// Creates a new `CircularBuffer` with the given capacity.
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

    /// Inserts a new element into the buffer and returns the oldest element if
    /// the buffer was already full or `None` otherwise.
    pub(crate) fn push(&mut self, value: T) -> Option<T> {
        let popped_value = if self.buffer.len() == self.capacity {
            self.pop()
        } else {
            None
        };
        self.buffer.push_back(value);

        popped_value
    }

    /// Removes and returns the oldest element from the buffer or `None` if the
    /// buffer was already empty.
    pub(crate) fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    /// Resets the buffer to its initial state.
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
    fn new() {
        let buffer: CircularBuffer<i32> = CircularBuffer::new(10);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn push() {
        let mut buffer: CircularBuffer<i32> = CircularBuffer::new(3);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn pop() {
        let mut buffer: CircularBuffer<i32> = CircularBuffer::new(3);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn next() {
        let mut buffer: CircularBuffer<i32> = CircularBuffer::new(3);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.push(3), None);
        assert_eq!(buffer.push(4), Some(1));
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn clear() {
        let mut buffer: CircularBuffer<i32> = CircularBuffer::new(3);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.push(3), None);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
    }
}
