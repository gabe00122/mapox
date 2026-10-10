use crate::envs::survival::ItemType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RefineType {
    Cook = 0,
}

impl RefineType {
    pub(super) fn refine(&self, input: ItemType) -> Option<ItemType> {
        REFINE_RECEPIES[*self as usize]
            .iter()
            .find(|(req, _)| *req == input)
            .map(|(_, out)| *out)
    }
}

pub(super) const REFINE_RECEPIES: &[&[(ItemType, ItemType)]] = &[
    &[(ItemType::Berry, ItemType::CookedBerry)], // cooking
];
