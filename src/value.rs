use derivative::Derivative;
use std::ops::{Add, Div, Mul, Sub};

/*
 * A wrapper around a numerical value, which
 * performs a derivative computation in the tangent mode.
 */

#[derive(Derivative)]
#[derivative(Debug, Clone, Copy, Default)]
pub struct Value {
    pub value: f32,
    #[derivative(Default(value = "0.0"))]
    pub der: f32,
}

impl Value {
    pub fn new(value: f32, der: f32) -> Self {
        Value { value, der }
    }

    pub fn pow(self, exp: f32) -> Self {
        let value = self.value.powf(exp);
        let der = exp * self.value.powf(exp - 1.0) * self.der;
        Value { value, der }
    }

    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let der = 0.5 * self.value.powf(-0.5) * self.der;
        Value { value, der }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let value = self.value + rhs.value;
        let der = self.der + rhs.der;
        Value { value, der }
    }
}

impl Add<&mut Value> for &mut Value {
    type Output = Value;
    fn add(self, rhs: &mut Value) -> Self::Output {
        let value = self.value + rhs.value;
        let der = self.der + rhs.der;
        Value { value, der }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        let value = self.value - rhs.value;
        let der = self.der - rhs.der;
        Value { value, der }
    }
}

impl Sub<&mut Value> for &mut Value {
    type Output = Value;
    fn sub(self, rhs: &mut Value) -> Self::Output {
        rhs.value = -1.0 * rhs.value;
        self + rhs
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let value = self.value * rhs.value;
        let der = rhs.value * self.der + self.value * rhs.der;
        Value { value, der }
    }
}

impl Mul<&mut Value> for &mut Value {
    type Output = Value;

    fn mul(self, rhs: &mut Value) -> Self::Output {
        let value = self.value * rhs.value;
        let der = rhs.value * self.der + self.value * rhs.der;
        Value { value, der }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, mut rhs: Value) -> Self::Output {
        rhs.der = -rhs.der / rhs.value / rhs.value;
        rhs.value = 1.0 / rhs.value;
        self * rhs
    }
}

impl Div<&mut Value> for &mut Value {
    type Output = Value;

    fn div(self, rhs: &mut Value) -> Self::Output {
        rhs.der = -rhs.der / rhs.value / rhs.value;
        rhs.value = 1.0 / rhs.value;
        self * rhs
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_default_value() {
        let x = Value::new(5.0, Default::default());

        assert_eq!(x.value, 5.0);
        assert_eq!(x.der, 0.0);
    }

    #[test]
    fn test_pow_operator() {
        let x = Value::new(3.0, 4.0);
        let y = x.pow(3.0);
        let z = x * x * x;

        assert_eq!(y.value, z.value);
        assert_eq!(y.der, z.der);
    }

    #[test]
    fn test_sqrt_operator() {
        let x = Value::new(1.5, 2.5);
        let y = x.sqrt();
        let z = x.pow(0.5);

        assert_eq!(y.value, z.value);
        assert_eq!(y.der, z.der);
    }

    fn f1(x: Value) -> Value {
        // f1(x)    = 2 * x^3
        // df1/dx   = 6 * x^2
        let a = Value::new(2.0, Default::default());

        a * x.pow(3.0)
    }

    fn f2(x: Value) -> Value {
        // f2(x)    = 2 / x^0.5
        // df2/dx   = -1 / x^1.5
        let a = Value::new(2.0, Default::default());

        a / x.sqrt()
    }

    fn f3(x: Value) -> Value {
        // f(x)     = 0.5 * x^3 + 1 / x
        // df3/dx   = 1.5 * x^2 - 1 / x^2
        let a = Value::new(0.5, Default::default());
        let b = Value::new(1.0, Default::default());

        a * x.pow(3.0) + b / x
    }

    #[test]
    fn test_derivative1() {
        let x = Value::new(2.0, 1.0);
        let y = f1(x);

        assert_eq!(y.value, 16.0);
        assert_eq!(y.der, 24.0);
    }

    #[test]
    fn test_derivative2() {
        let x = Value::new(4.0, 1.0);
        let y = f2(x);

        assert_eq!(y.value, 1.0);
        assert_eq!(y.der, -0.125);
    }

    #[test]
    fn test_derivative3() {
        let x = Value::new(2.0, 1.0);
        let y = f3(x);

        assert_eq!(y.value, 4.5);
        assert_eq!(y.der, 5.75);
    }
}
