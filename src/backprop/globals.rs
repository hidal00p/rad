use super::tape::GradientTape;
use std::sync::atomic::AtomicUsize;

pub static NAME_IDX: AtomicUsize = AtomicUsize::new(0); // AtomicUsize is thread-safe
pub static mut GRADIENT_TAPE: Option<GradientTape> = None;
