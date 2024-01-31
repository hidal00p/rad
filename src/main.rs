mod value;

use value::Value;

fn f(x: Value) -> Value {
    let a = Value::new(3.0, 0.0); // Passive variable
    let b = Value::new(2.0, 0.0); // Passive variable

    // purely copying
    a * x * x + b / x
}

fn univariate_example() {
    /*
     * Derivative of univariate f(x) = 3*x^2 + 2/x
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

fn g(x1: Value, x2: Value) -> Value {
    return x1 * x2;
}

fn multivariate_example() {
    /*
     * Derivative of multivariate g(x1, x2) = x1*x2
     * at (x1, x2) = (2.0, 3.0)
     *
     * g(x1 = 2.0, x2 = 3.0) = 6.0
     * ∇g = (∂g/∂x1, ∂g/∂x2) = (x2, x1) = (3.0, 2.0)
     *
     * This example is a little more involved, since it
     * requires an alternating seeding technique to extract a
     * Jacobian.
     *
     * Note that this could be extended to R^n.
     */
    let mut x = vec![Value::new(2.0, 0.0), Value::new(3.0, 0.0)]; // vector of arguments i.e. x = (x1, x2)
    let mut grad_g = vec![0.0; 2];
    for i in 0..2 {
        x[i].der = 1.0; // seed the desired partial

        let y = g(x[0], x[1]);
        grad_g[i] = y.der; // harvest the derivative

        x[i].der = 0.0; // clean-up
    }

    assert_eq!(grad_g[0], x[1].value);
    assert_eq!(grad_g[1], x[0].value);
}

fn main() {
    univariate_example();
    multivariate_example();
}
