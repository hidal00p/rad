use super::globals::NAME_IDX;
use super::variable::Variable;
use std::sync::atomic::Ordering;

pub struct GradientTape {
    pub entries: Vec<TapeEntry>,
}

impl GradientTape {
    pub fn new() -> Self {
        GradientTape {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: TapeEntry) {
        self.entries.push(entry);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        NAME_IDX.store(0, Ordering::SeqCst);
    }
}

pub trait CloneableFn: Fn(&Vec<Option<Variable>>) -> Vec<Variable> + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneableFn>;
}

impl<T> CloneableFn for T
where
    T: 'static + Fn(&Vec<Option<Variable>>) -> Vec<Variable> + Send + Sync + Clone,
{
    fn clone_box(&self) -> Box<dyn CloneableFn> {
        Box::new(self.clone())
    }
}

type GradientFunction = Box<dyn CloneableFn>;

pub struct TapeEntry {
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub propagate: GradientFunction,
}

impl Clone for TapeEntry {
    fn clone(&self) -> Self {
        TapeEntry {
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            propagate: self.propagate.clone_box(), // This needs a workaround
        }
    }
}

impl TapeEntry {
    pub fn new(inputs: Vec<Variable>, outputs: Vec<Variable>, propagate: GradientFunction) -> Self {
        TapeEntry {
            inputs,
            outputs,
            propagate,
        }
    }
}
