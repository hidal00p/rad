use std::collections::HashMap;
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

type GradientFunction = Box<dyn Fn(&Vec<Option<Variable>>) -> Vec<Variable>>;

pub struct TapeEntry {
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub propagate: GradientFunction,
}

impl TapeEntry {
    pub fn new(inputs: Vec<Variable>, outputs: Vec<Variable>, propagate: GradientFunction) -> Self {
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
        unsafe {
            NAME_IDX.store(0, Ordering::SeqCst);
        }
    }
}

impl Mul for Variable {
    type Output = Variable;

    fn mul(self, rhs: Variable) -> Self::Output {
        let result = Variable::new(self.value * rhs.value, None);
        println!("{} = {} * {}", self.value, rhs.value, result.value);

        let inputs = vec![self.clone(), rhs.clone()];
        let outputs = vec![result.clone()];

        let propagate = move |dloss_doutputs: &Vec<Option<Variable>>| -> Vec<Variable> {
            let dloss_dresult = dloss_doutputs.get(0).unwrap().clone().unwrap();

            let dresult_dself = rhs.value;
            let dresult_drhs = self.value;

            let dloss_dself = dloss_dresult.value * dresult_dself;
            let dloss_drhs = dloss_dresult.value * dresult_drhs;

            let dloss_dinputs = vec![
                Variable::new(dloss_dself, None),
                Variable::new(dloss_drhs, None),
            ];
            dloss_dinputs
        };

        unsafe {
            if let Some(tape) = &mut GRADIENT_TAPE {
                tape.add_entry(TapeEntry::new(inputs, outputs, Box::new(propagate)));
            };
        }
        result
    }
}

impl Add for Variable {
    type Output = Variable;

    fn add(self, rhs: Variable) -> Self::Output {
        todo!();
    }
}

impl Sub for Variable {
    type Output = Variable;

    fn sub(self, rhs: Variable) -> Self::Output {
        todo!();
    }
}

impl Div for Variable {
    type Output = Variable;

    fn div(self, rhs: Variable) -> Self::Output {
        todo!();
    }
}

impl Neg for Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        todo!();
    }
}

fn grad(loss: &Variable, desired_results: &Vec<Variable>) -> Vec<Option<Variable>> {
    let mut dloss_d = HashMap::new();
    dloss_d.insert(loss.name.clone(), Variable::new(1.0, None));

    fn gather_grad(
        entries: &Vec<Variable>,
        dloss_d: &HashMap<String, Variable>,
    ) -> Vec<Option<Variable>> {
        entries
            .iter()
            .map(|entry| {
                if let Some(dloss_dentry) = dloss_d.get(&entry.name) {
                    Some(dloss_dentry.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    println!("d{}:\n-----------", loss.name);
    unsafe {
        if let Some(tape) = &GRADIENT_TAPE {
            for entry in tape.entries.iter().rev() {
                let dloss_doutputs = gather_grad(&entry.outputs, &dloss_d);
                if dloss_doutputs.iter().all(|x| x.is_none()) {
                    continue;
                }

                let dloss_dinputs = (entry.propagate)(&dloss_doutputs);
                for (i, input) in entry.inputs.iter().enumerate() {
                    let dloss_dinput = dloss_dinputs.get(i);
                    if dloss_d.contains_key(&input.name) {
                        let current = dloss_d.get_mut(&input.name).unwrap();
                        current.value += dloss_dinput.unwrap().value;
                    } else {
                        dloss_d.insert(input.name.clone(), dloss_dinput.unwrap().clone());
                    }
                }
            }
        }
    }
    for (name, value) in &dloss_d {
        println!("d{}_d{} = {}", loss.name, name, value.value);
    }
    println!("-----------");

    gather_grad(desired_results, &dloss_d)
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

    #[test]
    fn test_mul_backprop() {
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        unsafe {
            GRADIENT_TAPE = Some(GradientTape::new());
        }
        let loss = a.clone() * b.clone();
        let da = grad(&loss, &vec![a]).get(0).unwrap().clone().unwrap();
        let db = grad(&loss, &vec![b]).get(0).unwrap().clone().unwrap();
        assert_eq!(da.value, 2.0);
        assert_eq!(db.value, 3.0);
    }
}
