/// Asserts that the given types are `Send`.
#[macro_export]
#[cfg(test)]
macro_rules! assert_send {
    ($($t:ty),+ $(,)?) => {
        fn assert_send<T: Send>() {}
        $(
            assert_send::<$t>();
        )+
    };
}

/// Asserts that the given types are `Sync`.
#[macro_export]
#[cfg(test)]
macro_rules! assert_sync {
    ($($t:ty),+ $(,)?) => {
        fn assert_sync<T: Sync>() {}
        $(
            assert_sync::<$t>();
        )+
    };
}
