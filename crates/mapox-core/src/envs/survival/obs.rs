//! What each agent sees: its field of view, darkened by night, and the UI
//! band of stats under it.

use ndarray::{Array2, ArrayViewMut2, s};

use crate::{
    envs::{
        common::{
            Position, UI_HEIGHT,
            fov::{self, ViewTile},
            ui::write_number,
        },
        survival::{
            Survival,
            needs::{Health, Hunger},
            world::Agent,
        },
    },
    symbols::{
        AGENT_GENERIC, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4,
        TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_MASK, TILE_UI, TILE_WALL, TILE_WATER, UI_DIGITS,
        UI_HEALTH, UI_HUNGER,
    },
    timestep::TimeStepMut,
    vocab::VocabId,
    vocab_enum,
};

vocab_enum!(pub(super) SurvivalObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    AgentGeneric => AGENT_GENERIC,
    UiHealth => UI_HEALTH,
    UiHunger => UI_HUNGER,
    Digit0 => UI_DIGITS[0],
    Digit1 => UI_DIGITS[1],
    Digit2 => UI_DIGITS[2],
    Digit3 => UI_DIGITS[3],
    Digit4 => UI_DIGITS[4],
    Digit5 => UI_DIGITS[5],
    Digit6 => UI_DIGITS[6],
    Digit7 => UI_DIGITS[7],
    Digit8 => UI_DIGITS[8],
    Digit9 => UI_DIGITS[9],
});

/// Where the UI band puts things. Its top row is the stats, each a label
/// then a three digit number: health from column 0, hunger from column 5.
const HEALTH_COL: usize = 0;
const HUNGER_COL: usize = 5;
const STAT_DIGITS: usize = 3;
/// The narrowest view the band fits in.
pub(super) const UI_WIDTH: usize = HUNGER_COL + 1 + STAT_DIGITS;

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

impl SurvivalObs {
    pub(super) fn move_blocked(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileWall | TileDestructibleWall | TileWater | AgentGeneric
        )
    }

    pub(super) fn spawnable(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }
}

impl ViewTile for SurvivalObs {
    const MASK: Self = SurvivalObs::Mask;

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use SurvivalObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }
}

impl Survival {
    pub(super) fn encode_observations(&mut self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        // the top row of the band is the stats
        let stats_row = fov_height + UI_HEIGHT - 1;

        for (&position, agent, health, hunger) in
            self.state
                .world
                .query_mut::<(&Position, &Agent, &Health, &Hunger)>()
        {
            let agent_id = agent.agent_index;

            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            encode_view(
                &self.state.render_map,
                &self.state.lighting,
                position,
                self.config.night_vision_radius,
                &mut view,
            );

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(SurvivalObs::UI.into());

            let mut stats = timestep.obs.slice_mut(s![agent_id, .., stats_row, 0]);
            for (col, label, value) in [
                (HEALTH_COL, SurvivalObs::UiHealth, health.amount),
                (HUNGER_COL, SurvivalObs::UiHunger, hunger.amount),
            ] {
                stats[col] = label.into();
                write_number(
                    stats.slice_mut(s![col + 1..col + 1 + STAT_DIGITS]),
                    u32::from(value),
                    &DIGIT_TILES,
                );
            }
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }
}

fn encode_view(
    map: &Array2<SurvivalObs>,
    lighting: &Array2<bool>,
    viewer: Position,
    night_vision_radius: i32,
    view: &mut ArrayViewMut2<VocabId>,
) {
    let (width, height) = view.dim();
    // the sweep runs over the window itself, with the viewer at its centre
    let center = Position::new(width as i32 / 2, height as i32 / 2);
    let origin = viewer - center;

    view.fill(SurvivalObs::MASK.into());
    fov::shadowcast(
        center,
        (width, height),
        |cell| !map[(origin + cell).idx()].opaque(),
        |cell| {
            let position = origin + cell;
            if lighting[position.idx()] || fov::within(cell - center, night_vision_radius) {
                view[cell.idx()] = map[position.idx()].into();
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::Environment, envs::survival::config::SurvivalConfig, timestep::TimeStepBuffers,
    };

    #[test]
    fn the_ui_band_shows_health_and_hunger() {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        let agent = env.state.agents[0];
        env.state.world.get::<&mut Health>(agent).unwrap().amount = 7;
        env.state.world.get::<&mut Hunger>(agent).unwrap().amount = 42;
        env.encode_observations(&mut buffers.view_mut());

        let fov = config.view_height as usize;
        let width = config.view_width as usize;
        let row = |y: usize| {
            (0..width)
                .map(|x| buffers.obs[[0, x, y, 0]])
                .collect::<Vec<_>>()
        };

        use SurvivalObs::*;
        let mut stats = vec![UI; width];
        stats[..9].copy_from_slice(&[UiHealth, UI, UI, Digit7, UI, UiHunger, UI, Digit4, Digit2]);
        let stats: Vec<VocabId> = stats.into_iter().map(Into::into).collect();
        assert_eq!(row(fov + UI_HEIGHT - 1), stats);
        assert_eq!(row(fov), vec![VocabId::from(UI); width]);
    }
}
