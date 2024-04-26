use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};

static NAME_IDX: AtomicUsize = AtomicUsize::new(0); // AtomicUsize is thread-safe
static mut GRADIENT_TAPE: Option<GradientTape> = None;

#[derive(Debug, Clone)]
pub struct Variable {
    pub value: f32,
    pub name: String,
}

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

type GradientFunction = Box<dyn Fn(&[Variable]) -> Vec<Variable>>;

pub struct TapeEntry {
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub propagate: GradientFunction,
}

impl TapeEntry {
    pub fn new(
        inputs: Vec<Variable>,
        outputs: Vec<Variable>,
        propagate: GradientFunction,
    ) -> Self {
        TapeEntry {
            inputs,
            outputs,
            propagate,
        }
    }
}

struct GradientTape {
    entries: Vec<TapeEntry>,
}

impl GradientTape {
    pub fn new() -> Self {
        GradientTape {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: TapeEntry) {
        self.entries.push(entry);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Mul for Variable {
    type Output = Variable;

    fn mul(self, rhs: Variable) -> Self::Output {
        let result = Variable::new(self.value * rhs.value, None);
        println!("{} = {} * {}", self.value, rhs.value, result.value);

        let inputs = vec![self.clone(), rhs.clone()];
        let outputs = vec![result.clone()];

        let propagate = move |dloss_doutputs: &[Variable]| -> Vec<Variable> {
            let dloss_dresult = dloss_doutputs.get(0).unwrap();

            let dresult_dself = rhs.value;
            let dresult_drhs = self.value;

            let dloss_dself = dloss_dresult.value * dresult_dself;
            let dloss_drhs = dloss_dresult.value * dresult_drhs;

            let dloss_dinputs = vec![Variable::new(dloss_dself, None), Variable::new(dloss_drhs, None)];
            dloss_dinputs
        };


        unsafe {
            if let Some(tape) = &mut GRADIENT_TAPE {
                tape.add_entry(TapeEntry {
                    inputs,
                    outputs,
                    propagate: Box::new(propagate),
                });
            }
        }
        result
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
