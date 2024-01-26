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

    fn f(x: Value) -> Value {
        // Test frunction:
        //  f(x)     = 0.5 * x^3 + 1 / x
        //  df/dx    = 1.5 x^2 - 1 / x^2
        let a = Value::new(0.5, Default::default());
        let b = Value::new(1.0, Default::default());

        a * x * x * x + b / x
    }

    #[test]
    fn test_derivative() {
        let x = Value::new(2.0, 1.0); // x = 2, dx/dx = 1
        let y = f(x);

        assert_eq!(y.value, 4.5);
        assert_eq!(y.der, 5.75);
    }
}
