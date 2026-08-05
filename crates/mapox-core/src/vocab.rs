use std::collections::HashMap;

/// Integer representation used for vocabulary entries in maps and observations.
pub type VocabId = u8;

#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    symbols: Vec<&'static str>,
    ids: HashMap<&'static str, VocabId>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn add(&mut self, symbol: &'static str) -> VocabId {
        if let Some(&id) = self.ids.get(symbol) {
            return id;
        }

        let new_id =
            VocabId::try_from(self.symbols.len()).expect("vocabulary exhausted VocabId capacity");
        self.symbols.push(symbol);
        self.ids.insert(symbol, new_id);
        new_id
    }

    pub fn symbols(&self) -> &[&'static str] {
        &self.symbols
    }

    pub fn get(&self, symbol: &str) -> Option<VocabId> {
        self.ids.get(symbol).copied()
    }

    pub fn lut_to(&self, target: &Vocabulary, default: VocabId) -> Vec<VocabId> {
        self.symbols
            .iter()
            .map(|s| target.get(s).unwrap_or(default))
            .collect()
    }
}

impl PartialEq for Vocabulary {
    fn eq(&self, other: &Self) -> bool {
        self.symbols == other.symbols
    }
}

impl Eq for Vocabulary {}

impl Extend<&'static str> for Vocabulary {
    fn extend<T: IntoIterator<Item = &'static str>>(&mut self, iter: T) {
        for s in iter {
            self.add(s);
        }
    }
}

impl FromIterator<&'static str> for Vocabulary {
    fn from_iter<T: IntoIterator<Item = &'static str>>(iter: T) -> Self {
        let mut vocab = Self::new();
        vocab.extend(iter);
        vocab
    }
}
