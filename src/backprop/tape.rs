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

type GradientFunction = Box<dyn Fn(&Vec<Option<Variable>>) -> Vec<Variable>>;

pub struct TapeEntry {
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub propagate: GradientFunction,
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
