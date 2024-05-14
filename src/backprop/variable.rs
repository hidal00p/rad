use super::globals::{GRADIENT_TAPE, NAME_IDX};
use super::tape::TapeEntry;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, PartialEq)]
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

unsafe impl Send for Variable {}
unsafe impl Sync for Variable {}

impl Add for Variable {
    type Output = Variable;

    fn add(self, rhs: Variable) -> Self::Output {
        let result = Variable::new(self.value + rhs.value, None);
        println!("{} = {} + {} = {} + {} = {}", result.name, self.name, rhs.name, self.value, rhs.value, result.value);

        let inputs = vec![self.clone(), rhs.clone()];
        let outputs = vec![result.clone()];

        let propagate = move |dloss_doutputs: &Vec<Option<Variable>>| -> Vec<Variable> {
            let dloss_dresult = dloss_doutputs.get(0).unwrap().clone().unwrap();

            let dresult_dself = Variable::new(1.0, None);
            let dresult_drhs = Variable::new(1.0, None);

            let dloss_dself = dloss_dresult.clone() * dresult_dself;
            let dloss_drhs = dloss_dresult * dresult_drhs;

            let dloss_dinputs = vec![dloss_dself, dloss_drhs];
            dloss_dinputs
        };

        let tape_entry = TapeEntry::new(inputs, outputs, Box::new(propagate));
        GRADIENT_TAPE.lock().unwrap().add_entry(tape_entry);

        result
    }
}

impl Sub for Variable {
    type Output = Variable;

    fn sub(self, rhs: Variable) -> Self::Output {
        let result = Variable::new(self.value - rhs.value, None);
        println!("{} = {} - {} = {} - {} = {}", result.name, self.name, rhs.name, self.value, rhs.value, result.value);

        let inputs = vec![self.clone(), rhs.clone()];
        let outputs = vec![result.clone()];

        let propagate = move |dloss_doutputs: &Vec<Option<Variable>>| -> Vec<Variable> {
            let dloss_dresult = dloss_doutputs.get(0).unwrap().clone().unwrap();

            let dresult_dself = Variable::new(1.0, None);
            let dresult_drhs = Variable::new(-1.0, None);

            let dloss_dself = dloss_dresult.clone() * dresult_dself;
            let dloss_drhs = dloss_dresult * dresult_drhs;

            let dloss_dinputs = vec![dloss_dself, dloss_drhs];
            dloss_dinputs
        };

        let tape_entry = TapeEntry::new(inputs, outputs, Box::new(propagate));
        GRADIENT_TAPE.lock().unwrap().add_entry(tape_entry);

        result
    }
}

impl Mul for Variable {
    type Output = Variable;

    fn mul(self, rhs: Variable) -> Self::Output {
        let result = Variable::new(self.value * rhs.value, None);
        println!("{} = {} * {} = {} * {} = {}", result.name, self.name, rhs.name, self.value, rhs.value, result.value);

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

        let tape_entry = TapeEntry::new(inputs, outputs, Box::new(propagate));
        GRADIENT_TAPE.lock().unwrap().add_entry(tape_entry);

        result
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
        Variable::new(-1.0, None) * self
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
        NAME_IDX.store(0, Ordering::SeqCst);
        let v0 = Variable::new(3.0, None);
        let v1 = Variable::new(3.0, None);
        assert_eq!(v0.name, "v0");
        assert_eq!(v1.name, "v1");
    }

    #[test]
    fn test_simple_add() {
        let a = Variable::new(2.0, None);
        let b = Variable::new(3.0, None);
        let c = a + b;
        assert_eq!(c.value, 5.0);
    }

    #[test]
    fn test_simple_sub() {
        let a = Variable::new(2.5, None);
        let b = Variable::new(0.5, None);
        let c = a - b;
        assert_eq!(c.value, 2.0);
    }

    #[test]
    fn test_simple_mul() {
        let a = Variable::new(2.0, None);
        let b = Variable::new(3.0, None);
        let c = a * b;
        assert_eq!(c.value, 6.0);
    }

    #[test]
    fn test_simple_div() {
        let a = Variable::new(12.0, None);
        let b = Variable::new(3.0, None);
        let c = a / b;
        assert_eq!(c.value, 4.0);
    }

    #[test]
    fn test_simple_neg() {
        let a = Variable::new(3.0, None);
        let b = -a;
        assert_eq!(b.value, -3.0);
    }
}
