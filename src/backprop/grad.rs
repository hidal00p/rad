use super::globals::GRADIENT_TAPE;
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

    let entries = {
        let tape = GRADIENT_TAPE.lock().unwrap();
        tape.entries.clone()
    };

    println!("d{}:\n-----------", loss.name);
    for entry in &entries {
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
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() + b.clone();
        let dloss_d = grad(&loss, &vec![a, b]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        let dloss_db = dloss_d.get(1).unwrap().clone().unwrap();
        assert_eq!(dloss_da.value, 1.0);
        assert_eq!(dloss_db.value, 1.0);
    }

    #[test]
    fn test_backprop_simple_sub() {
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() - b.clone();
        let dloss_d = grad(&loss, &vec![a, b]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        let dloss_db = dloss_d.get(1).unwrap().clone().unwrap();
        assert_eq!(dloss_da.value, 1.0);
        assert_eq!(dloss_db.value, -1.0);
    }

    #[test]
    fn test_backprop_simple_mul() {
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() * b.clone();
        let dloss_d = grad(&loss, &vec![a, b]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        let dloss_db = dloss_d.get(1).unwrap().clone().unwrap();
        assert_eq!(dloss_da.value, 2.0);
        assert_eq!(dloss_db.value, 3.0);
    }

    #[test]
    fn test_backprop_simple_div() {
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() / b.clone();
        let dloss_d = grad(&loss, &vec![a, b]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        let dloss_db = dloss_d.get(1).unwrap().clone().unwrap();
        assert_eq!(dloss_da.value, 0.5);
        assert_eq!(dloss_db.value, -0.75);
    }

    #[test]
    fn test_backprop_simple_neg() {
        let a = Variable::new(3.0, None);
        let loss = -a.clone();
        let dloss_d = grad(&loss, &vec![a]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        assert_eq!(dloss_da.value, -1.0);
    }

    #[test]
    fn test_backprop_zero_grad() {
        let a = Variable::new(3.0, Some('a'.to_string()));
        let b = Variable::new(2.0, Some('b'.to_string()));
        let loss = a.clone() * a.clone();
        let dloss_d = grad(&loss, &vec![a, b]);
        let dloss_da = dloss_d.get(0).unwrap().clone().unwrap();
        let dloss_db = dloss_d.get(1);
        assert_eq!(dloss_da.value, 6.0);
        assert_eq!(dloss_db, Some(None).as_ref());
    }
}
