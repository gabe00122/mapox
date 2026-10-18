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
            items::Inventory,
            needs::{Health, Hunger},
            world::Agent,
        },
    },
    symbols::{
        AGENT_GENERIC_DOWN, AGENT_GENERIC_LEFT, AGENT_GENERIC_RIGHT, AGENT_GENERIC_UP, ITEM_APPLE,
        ITEM_BERRY, ITEM_CAMPFIRE_KIT, ITEM_COOKED_BERRY, ITEM_CUT_GRASS, ITEM_IRON_ORE, ITEM_ROCK,
        ITEM_STICK, ITEM_STONE_AXE, ITEM_STONE_PICKAXE, ITEM_WOOD, TILE_APPLE_TREE,
        TILE_BERRY_BUSH, TILE_BUSH, TILE_CUT_GRASS, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3,
        TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_FIRE, TILE_FIRE_OUT, TILE_IRON_ORE,
        TILE_MASK, TILE_PORTAL, TILE_TALL_GRASS, TILE_THORN_BUSH, TILE_TREE, TILE_UI, TILE_WALL,
        TILE_WATER, UI_DIGITS, UI_HEALTH, UI_HUNGER,
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
    TileIronOre => TILE_IRON_ORE,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    TileFire => TILE_FIRE,
    TileFireOut => TILE_FIRE_OUT,
    TileTree => TILE_TREE,
    TileAppleTree => TILE_APPLE_TREE,
    TileBerryBush => TILE_BERRY_BUSH,
    TileBush => TILE_BUSH,
    TileThornBush => TILE_THORN_BUSH,
    TileTallGrass => TILE_TALL_GRASS,
    TileCutGrass => TILE_CUT_GRASS,
    TilePortal => TILE_PORTAL,
    AgentUp => AGENT_GENERIC_UP,
    AgentRight => AGENT_GENERIC_RIGHT,
    AgentDown => AGENT_GENERIC_DOWN,
    AgentLeft => AGENT_GENERIC_LEFT,
    ItemRock => ITEM_ROCK,
    ItemStick => ITEM_STICK,
    ItemWood => ITEM_WOOD,
    ItemCutGrass => ITEM_CUT_GRASS,
    ItemIronOre => ITEM_IRON_ORE,
    ItemStoneAxe => ITEM_STONE_AXE,
    ItemStonePickaxe => ITEM_STONE_PICKAXE,
    ItemCampfireKit => ITEM_CAMPFIRE_KIT,
    ItemBerry => ITEM_BERRY,
    ItemCookedBerry => ITEM_COOKED_BERRY,
    ItemApple => ITEM_APPLE,
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
/// then a two digit number: health from column 0, hunger from column 4.
const HEALTH_COL: usize = 0;
const HUNGER_COL: usize = 4;
const STAT_DIGITS: usize = 2;
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
            TileWall
                | TileDestructibleWall
                | TileIronOre
                | TileWater
                | AgentUp
                | AgentRight
                | AgentDown
                | AgentLeft
        )
    }
}

impl ViewTile for SurvivalObs {
    const MASK: Self = SurvivalObs::Mask;

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use SurvivalObs::*;
        matches!(self, TileWall | TileDestructibleWall | TileIronOre)
    }
}

impl Survival {
    pub(super) fn encode_observations(&mut self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        timestep.obs.fill(SurvivalObs::Mask.into());
        timestep.reward.fill(0.0);
        timestep.terminated.fill(true);

        for (&position, agent, health, hunger, inventory) in
            self.state
                .world
                .query_mut::<(&Position, &mut Agent, &Health, &Hunger, &Inventory)>()
        {
            let agent_index = agent.agent_index;

            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_index, .., ..fov_height, 0]);
            encode_view(
                &self.state.render_map,
                &self.state.lighting,
                position,
                self.config
                    .night_vision_radius
                    .max(self.state.dusk.unwrap_or(0)),
                &mut view,
            );

            let mut ui = timestep.obs.slice_mut(s![agent_index, .., fov_height.., 0]);
            ui.fill(SurvivalObs::UI.into());

            // draw stats (the last row of the UI band)
            let mut stats = ui.slice_mut(s![.., UI_HEIGHT - 1]);
            for (col, label, value) in [
                (HEALTH_COL, SurvivalObs::UiHealth, health.amount),
                (HUNGER_COL, SurvivalObs::UiHunger, hunger.amount),
            ] {
                stats[col] = label.into();
                write_number(
                    stats.slice_mut(s![col + 1..col + 1 + STAT_DIGITS]),
                    value.max(0).cast_unsigned(),
                    &DIGIT_TILES,
                );
            }

            for (i, slot) in inventory.slots.iter().enumerate() {
                ui[(i, 0)] = if let Some(slot) = slot {
                    slot.obs().into()
                } else {
                    SurvivalObs::TileEmpty.into()
                }
            }

            timestep.reward[agent_index] = agent.reward;
            timestep.terminated[agent_index] = health.amount == 0 || self.state.time == self.length;
            agent.reward = 0.0;
        }

        timestep.time.fill(self.state.time as i32);
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
        stats[..7].copy_from_slice(&[UiHealth, UI, Digit7, UI, UiHunger, Digit4, Digit2]);
        let stats: Vec<VocabId> = stats.into_iter().map(Into::into).collect();
        assert_eq!(row(fov + UI_HEIGHT - 1), stats);
        assert_eq!(row(fov), vec![VocabId::from(UI); width]);
    }
}
