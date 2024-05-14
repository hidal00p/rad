use super::tape::GradientTape;
use once_cell::sync::Lazy;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

pub static NAME_IDX: AtomicUsize = AtomicUsize::new(0); // AtomicUsize is thread-safe
pub static GRADIENT_TAPE: Lazy<Arc<Mutex<GradientTape>>> =
    Lazy::new(|| Arc::new(Mutex::new(GradientTape::new())));
