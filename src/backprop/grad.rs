use super::globals::GRADIENT_TAPE;
use super::tape::GradientTape;
use super::variable::Variable;
use std::collections::HashMap;

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
    fn test_backprop_simple_add() {
        unsafe {
            GRADIENT_TAPE = Some(GradientTape::new());
        }
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() + b.clone();
        let da = grad(&loss, &vec![a]).get(0).unwrap().clone().unwrap();
        let db = grad(&loss, &vec![b]).get(0).unwrap().clone().unwrap();
        assert_eq!(da.value, 1.0);
        assert_eq!(db.value, 1.0);
    }

    #[test]
    fn test_backprop_simple_sub() {
        unsafe {
            GRADIENT_TAPE = Some(GradientTape::new());
        }
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() - b.clone();
        let da = grad(&loss, &vec![a]).get(0).unwrap().clone().unwrap();
        let db = grad(&loss, &vec![b]).get(0).unwrap().clone().unwrap();
        assert_eq!(da.value, 1.0);
        assert_eq!(db.value, -1.0);
    }

    #[test]
    fn test_backprop_simple_mul() {
        unsafe {
            GRADIENT_TAPE = Some(GradientTape::new());
        }
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() * b.clone();
        let da = grad(&loss, &vec![a]).get(0).unwrap().clone().unwrap();
        let db = grad(&loss, &vec![b]).get(0).unwrap().clone().unwrap();
        assert_eq!(da.value, 2.0);
        assert_eq!(db.value, 3.0);
    }
}
