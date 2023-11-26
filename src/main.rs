mod value;

use crate::value::Value;

fn f1(x0: &mut Value, x1: &mut Value) -> Value {
    x0 * x1
}

fn f2(x0: &mut Value, x1: &mut Value) -> Value {
    x0 / x1
}

fn main() {
    let mut a: Value = Value::new(2.0, 0.0);
    let mut b: Value = Value::new(3.0, 1.0);

    println!("{:?}", f1(&mut a, &mut b));
    println!("{:?}", f2(&mut a, &mut b));
}
