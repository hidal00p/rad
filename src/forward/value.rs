use std::ops::{Add, Div, Mul, Neg, Sub};

/*
 * A wrapper around a numerical value, which
 * performs a derivative computation in the tangent mode.
 */

#[derive(Debug, Clone, Copy)]
pub struct Value {
    pub value: f32,
    pub der: f32,
}

impl Value {
    pub fn passive(value: f32) -> Self {
        Value { value, der: 0.0 }
    }

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

    pub fn relu(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Value::new(0.0, 0.0)
        }
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

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        let value = self.value - rhs.value;
        let der = self.der - rhs.der;
        Value { value, der }
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

impl Div for Value {
    type Output = Value;

    fn div(self, mut rhs: Value) -> Self::Output {
        rhs.der = -rhs.der / rhs.value / rhs.value;
        rhs.value = 1.0 / rhs.value;
        self * rhs
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        let value = -self.value;
        let der = -self.der;
        Value { value, der }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_default_value() {
        let x = Value::passive(5.0);

        assert_eq!(x.value, 5.0);
        assert_eq!(x.der, 0.0);
    }

    #[test]
    fn test_neg() {
        let x = -Value::new(1.0, 2.0);

        assert_eq!(x.value, -1.0);
        assert_eq!(x.der, -2.0);
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

    #[test]
    fn test_pos_relu_operator() {
        let x = Value::new(1.5, 2.5);
        let y = x.relu();

        assert_eq!(y.value, x.value);
        assert_eq!(y.der, x.der);
    }

    #[test]
    fn test_neg_relu_operator() {
        let x = Value::new(-1.5, 2.5);
        let y = x.relu();

        assert_eq!(y.value, 0.0);
        assert_eq!(y.der, 0.0);
    }

    #[test]
    fn test_zero_relu_operator() {
        let x = Value::new(0.0, 2.5);
        let y = x.relu();

        assert_eq!(y.value, 0.0);
        assert_eq!(y.der, 0.0);
    }

    fn f1(x: Value) -> Value {
        // f1(x)    = 2 * x^3
        // f1'(x)   = 6 * x^2
        let a = Value::passive(2.0);

        a * x.pow(3.0)
    }

    #[test]
    fn test_differentiate_f1() {
        let x = Value::new(2.0, 1.0);
        let y = f1(x);

        assert_eq!(y.value, 16.0);
        assert_eq!(y.der, 24.0);
    }

    fn f2(x: Value) -> Value {
        // f2(x)    = 2 / x^0.5
        // f2'(x)   = -1 / x^1.5
        let a = Value::passive(2.0);

        a / x.sqrt()
    }

    #[test]
    fn test_differentiate_f2() {
        let x = Value::new(4.0, 1.0);
        let y = f2(x);

        assert_eq!(y.value, 1.0);
        assert_eq!(y.der, -0.125);
    }

    #[test]
    fn test_differentiate_sum() {
        let x = Value::new(4.0, 1.0);
        let y = f1(x) + f2(x);

        assert_eq!(y.value, 129.0);
        assert_eq!(y.der, 95.875);
    }

    #[test]
    fn test_differentiate_subraction() {
        let x = Value::new(4.0, 1.0);
        let y = f1(x) - f2(x);

        assert_eq!(y.value, 127.0);
        assert_eq!(y.der, 96.125);
    }

    #[test]
    fn test_differentiate_product() {
        // f3(x)    = f1(x) * f2(x) = 4 * x^2.5
        // f3'(x)   = 10 * x^1.5
        let x = Value::new(2.0, 1.0);
        let y = f1(x) * f2(x);

        assert_eq!(y.value, 22.627417);
        assert_eq!(y.der, 28.28427);
    }

    #[test]
    fn test_differentiate_chain() {
        // f4(x)    = f1(f2(x)) = 2 * ( 2 / x^-0.5 )^3
        //          = 16 / x^1.5
        // f4'(x)   = -24 / x^-2.5
        let x = Value::new(2.0, 1.0);
        let y = f1(f2(x));

        assert_eq!(y.value, 5.656854249);
        assert_eq!(y.der, -4.242640687);
    }
}
