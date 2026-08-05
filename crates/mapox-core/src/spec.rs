use crate::vocab::VocabId;

#[derive(Debug, Default, Clone, Copy)]
pub struct ActionSpec {
    pub num_actions: usize,
}

impl ActionSpec {
    pub fn new(num_actions: usize) -> Self {
        Self { num_actions }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ObservationSpec {
    pub width: i32,
    pub height: i32,
    pub num_types: usize,
}

impl ObservationSpec {
    pub fn new(width: i32, height: i32, num_types: usize) -> Self {
        Self {
            width,
            height,
            num_types,
        }
    }
}

pub fn action_mask_into(action_ids: impl IntoIterator<Item = VocabId>, mask: &mut [u8]) {
    for m in mask.iter_mut() {
        *m = 0;
    }

    for action_id in action_ids {
        mask[usize::from(action_id)] = 1
    }
}
