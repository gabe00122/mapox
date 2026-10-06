//! What each agent sees: its field of view, darkened by night, and the UI
//! band of stats and inventory under it.

use ndarray::s;

use crate::{
    envs::common::{Position, UI_HEIGHT, fov, ui::write_number},
    timestep::TimeStepMut,
};

use super::{
    Survival,
    tiles::{SurvivalObs, within},
};

/// Where the UI band puts things. Its top row is the stats, each a label
/// then a three digit number: health from column 0, hunger from column 5,
/// and the columns after that kept free for temperature. The row under it is
/// the inventory, each slot a label then the item, hands from column 0 and
/// backpack from column 3; then the sun or the moon at column 6.
pub(super) const HEALTH_COL: usize = 0;
pub(super) const HUNGER_COL: usize = 5;
pub(super) const STAT_DIGITS: usize = 3;
pub(super) const HANDS_COL: usize = 0;
pub(super) const BACKPACK_COL: usize = 3;
pub(super) const CLOCK_COL: usize = 6;
/// The narrowest view the band fits in.
pub(super) const UI_WIDTH: usize = HUNGER_COL + 1 + STAT_DIGITS;
const _: () = assert!(CLOCK_COL < UI_WIDTH);

const DIGIT_TILES: [SurvivalObs; 10] = [
    SurvivalObs::Digit0,
    SurvivalObs::Digit1,
    SurvivalObs::Digit2,
    SurvivalObs::Digit3,
    SurvivalObs::Digit4,
    SurvivalObs::Digit5,
    SurvivalObs::Digit6,
    SurvivalObs::Digit7,
    SurvivalObs::Digit8,
    SurvivalObs::Digit9,
];

impl Survival {
    pub(super) fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        // the top row of the band is the stats, the one under it inventory
        let stats_row = fov_height + UI_HEIGHT - 1;
        let slots_row = fov_height;
        let night = self.is_night(self.state.time);
        let sight = self.vision_radius(self.state.time);
        // as `fov::encode_visible` centres the window
        let half_width = self.config.view_width / 2;
        let half_height = self.config.view_height / 2;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::encode_visible(
                &self.state.map,
                agent.position,
                &mut view,
                SurvivalObs::Mask,
                |tile| tile.opaque(),
            );
            if let Some(sight) = sight {
                // only the near and the lit show in the dark
                let corner = agent.position - Position::new(half_width, half_height);
                for ((x, y), cell) in view.indexed_iter_mut() {
                    let position = corner + Position::new(x as i32, y as i32);
                    let near = within(position - agent.position, sight);
                    if !near && !self.state.lit[position.idx()] {
                        *cell = SurvivalObs::Mask.into();
                    }
                }
            }

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(SurvivalObs::UI.into());

            let mut stats = timestep.obs.slice_mut(s![agent_id, .., stats_row, 0]);
            for (col, label, value) in [
                (HEALTH_COL, SurvivalObs::UiHealth, agent.health),
                (HUNGER_COL, SurvivalObs::UiHunger, agent.hunger),
            ] {
                stats[col] = label.into();
                write_number(
                    stats.slice_mut(s![col + 1..col + 1 + STAT_DIGITS]),
                    u32::from(value),
                    &DIGIT_TILES,
                );
            }

            let mut slots = timestep.obs.slice_mut(s![agent_id, .., slots_row, 0]);
            for (col, label, item) in [
                (HANDS_COL, SurvivalObs::UiHands, agent.hands),
                (BACKPACK_COL, SurvivalObs::UiBackpack, agent.backpack),
            ] {
                slots[col] = label.into();
                if let Some(item) = item {
                    slots[col + 1] = item.tile().into();
                }
            }
            slots[CLOCK_COL] = if night {
                SurvivalObs::UiNight
            } else {
                SurvivalObs::UiDay
            }
            .into();
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }
}
