use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG: AtomicBool = AtomicBool::new(false);

/// Enable debug logging
pub fn enable() {
    DEBUG.store(true, Ordering::Relaxed);
}

/// Check if debug logging is enabled
pub fn is_enabled() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

/// Print debug message if debug mode is enabled
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::debug::is_enabled() {
            eprintln!($($arg)*);
        }
    };
}
