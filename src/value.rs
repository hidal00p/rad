use std::ops::{Add, Div, Mul, Sub};

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
    pub fn new(value: f32, der: f32) -> Self {
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
