use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct Variable {
    pub value: f32,
    pub name: String,
}

static NAME_IDX: AtomicUsize = AtomicUsize::new(0); // AtomicUsize is thread-safe

impl Variable {
    pub fn new(value: f32, name: Option<String>) -> Self {
        let name = match name {
            Some(n) => n,
            None => {
                let idx = NAME_IDX.fetch_add(1, Ordering::SeqCst); // SeqCst is the most stringent memory ordering ensuring thread-thread-safety
                format!("v{}", idx)
            }
        };

        Variable { value, name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable() {
        let x = Variable::new(3.0, Some("x".to_string()));
        assert_eq!(x.value, 3.0);
        assert_eq!(x.name, "x");
    }

    #[test]
    fn test_auto_name_generation() {
        let v0 = Variable::new(3.0, None);
        let v1 = Variable::new(3.0, None);
        assert_eq!(v0.name, "v0");
        assert_eq!(v1.name, "v1");
    }
}
