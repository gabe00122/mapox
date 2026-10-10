use crate::envs::survival::SurvivalObs;

pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
}

impl ItemType {
    pub(super) fn obs(&self) -> SurvivalObs {
        SurvivalObs::Digit0
    }
}

#[derive(Default)]
pub(super) struct Inventory {
    pub hand: Option<ItemType>,
    pub back: Option<ItemType>,
}
