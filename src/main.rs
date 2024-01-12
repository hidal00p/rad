mod value;

use value::Value;

fn f(x: Value) -> Value {
    let a = Value::new(3.0, 0.0); // Passive variable
    let b = Value::new(2.0, 0.0); // Passive variable

    // purely copying
    a * x * x + b / x
}

fn main() {
    /*
     * Derivative of f(x) = 3*x^2 + 2/x
     * at x = 2.0.
     *
     * f(x = 2.0) = 13.0
     * dfdx(x = 2.0) = 11.5
     */

    let x = Value::new(2.0, 1.0); // Active variable
    let y = f(x);

    assert_eq!(y.value, 13.0);
    assert_eq!(y.der, 11.5);
}
