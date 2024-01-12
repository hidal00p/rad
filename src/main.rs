mod value;

use value::Value;

fn f(mut x: Value) -> Value {
    let mut a = Value::new(3.0, 0.0);
    let mut b = Value::new(2.0, 0.0);

    (&mut a)*(&mut (x*x)) + (&mut b) / (&mut x)
}

fn main() {
    /*
     * Derivative of f(x) = 3*x^2 + 2/x
     * at x = 2.0.
     *
     * f(x = 2.0) = 13.0
     * dfdx(x = 2.0) = 11.5
     */

    let x = Value::new(2.0, 1.0);
    let y = f(x);

    assert_eq!(y.value, 13.0);
    assert_eq!(y.der, 11.5);
}
