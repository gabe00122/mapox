pub mod app;
pub mod env;
mod gpu;
mod grid;
pub mod keys;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod rgb;
pub mod tileset;

#[cfg(not(target_arch = "wasm32"))]
pub use app::open_window;
pub use app::{PacingMode, RenderApp, ViewMode};
pub use keys::{Command, Input};

use crate::{symbols, vocab::Vocabulary};

/// Symbol -> (col, row) on the sheet `scripts/make_tileset.py` draws. Row 0
/// is terrain, row 1 items and characters, rows 2 and 3 the soldiers, row 4
/// the UI digits and stat labels, row 5 pac-man, and rows 6 and 7 survival.
const TILE_ART: &[(&str, u32, u32)] = &[
    (symbols::TILE_UI, 0, 0),
    (symbols::TILE_MASK, 1, 0),
    (symbols::TILE_EMPTY, 2, 0),
    (symbols::TILE_WALL, 3, 0),
    (symbols::TILE_DESTRUCTIBLE_WALL, 4, 0),
    (symbols::TILE_WATER, 5, 0),
    (symbols::TILE_PIPE_HORIZONTAL, 6, 0),
    (symbols::TILE_PIPE_VIRTICAL, 7, 0),
    (symbols::TILE_DECOR_1, 8, 0),
    (symbols::TILE_DECOR_2, 9, 0),
    (symbols::TILE_DECOR_3, 10, 0),
    (symbols::TILE_DECOR_4, 11, 0),
    // Flags are treasure chests: shut and dull while locked, open and full of
    // gold once claimable.
    (symbols::TILE_FLAG, 0, 1),
    (symbols::TILE_FLAG_UNLOCKED, 1, 1),
    (symbols::ITEM_APPLE, 2, 1),
    (symbols::AGENT_GENERIC, 7, 1),
    (symbols::AGENT_SCOUT, 8, 1),
    (symbols::AGENT_HARVESTER, 9, 1),
    // Hide-and-seek reuses the king-of-the-hill soldiers: the unwounded
    // knight of each facing, blue for hiders and red for seekers.
    (symbols::AGENT_SEEKER_UP, 1, 2),
    (symbols::AGENT_SEEKER_RIGHT, 3, 2),
    (symbols::AGENT_SEEKER_DOWN, 5, 2),
    (symbols::AGENT_SEEKER_LEFT, 7, 2),
    (symbols::AGENT_HIDER_UP, 1, 3),
    (symbols::AGENT_HIDER_RIGHT, 3, 3),
    (symbols::AGENT_HIDER_DOWN, 5, 3),
    (symbols::AGENT_HIDER_LEFT, 7, 3),
    (symbols::UI_DIGITS[0], 0, 4),
    (symbols::UI_DIGITS[1], 1, 4),
    (symbols::UI_DIGITS[2], 2, 4),
    (symbols::UI_DIGITS[3], 3, 4),
    (symbols::UI_DIGITS[4], 4, 4),
    (symbols::UI_DIGITS[5], 5, 4),
    (symbols::UI_DIGITS[6], 6, 4),
    (symbols::UI_DIGITS[7], 7, 4),
    (symbols::UI_DIGITS[8], 8, 4),
    (symbols::UI_DIGITS[9], 9, 4),
    (symbols::UI_HEALTH, 10, 4),
    (symbols::UI_HUNGER, 11, 4),
    (symbols::TILE_PELLET, 0, 5),
    (symbols::TILE_POWER_PELLET, 1, 5),
    (symbols::AGENT_PACMAN_UP, 2, 5),
    (symbols::AGENT_PACMAN_RIGHT, 3, 5),
    (symbols::AGENT_PACMAN_DOWN, 4, 5),
    (symbols::AGENT_PACMAN_LEFT, 5, 5),
    (symbols::AGENT_GHOST_PINKY, 6, 5),
    (symbols::AGENT_GHOST_BLINKY, 7, 5),
    (symbols::AGENT_GHOST_INKY, 8, 5),
    (symbols::AGENT_GHOST_CLYDE, 9, 5),
    (symbols::AGENT_GHOST_FRIGHTENED, 10, 5),
    (symbols::AGENT_GHOST_EYES, 11, 5),
    (symbols::TILE_FIRE, 0, 6),
    (symbols::AGENT_GENERIC_UP, 1, 6),
    (symbols::AGENT_GENERIC_RIGHT, 2, 6),
    (symbols::AGENT_GENERIC_DOWN, 3, 6),
    (symbols::AGENT_GENERIC_LEFT, 4, 6),
    (symbols::ITEM_ROCK, 5, 6),
    (symbols::ITEM_STICK, 6, 6),
    (symbols::ITEM_WOOD, 7, 6),
    (symbols::ITEM_CUT_GRASS, 8, 6),
    (symbols::ITEM_STONE_AXE, 9, 6),
    (symbols::TILE_FIRE_OUT, 10, 6),
    (symbols::TILE_TREE, 11, 6),
    (symbols::TILE_BERRY_BUSH, 0, 7),
    (symbols::TILE_BUSH, 1, 7),
    (symbols::ITEM_BERRY, 2, 7),
    (symbols::ITEM_COOKED_BERRY, 3, 7),
    (symbols::ITEM_CAMPFIRE_KIT, 4, 7),
    (symbols::TILE_TALL_GRASS, 5, 7),
    (symbols::TILE_CUT_GRASS, 6, 7),
    (symbols::TILE_PORTAL, 7, 7),
    (symbols::TILE_APPLE_TREE, 8, 7),
    (symbols::TILE_THORN_BUSH, 9, 7),
    (symbols::ITEM_STONE_PICKAXE, 10, 7),
    (symbols::TILE_IRON_ORE, 11, 7),
    (symbols::ITEM_IRON_ORE, 11, 1),
];

pub(crate) fn resolve_art(vocab: &Vocabulary) -> Vec<(u32, u32)> {
    vocab
        .symbols()
        .iter()
        .map(|symbol| {
            TILE_ART
                .iter()
                .find(|(name, _, _)| name == symbol)
                .map(|&(_, col, row)| (col, row))
                .expect("every obs symbol has tile art")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use crate::envs::hide_seek::{HideSeek, HideSeekConfig};
    use crate::envs::maze::{Maze, MazeConfig};
    use crate::envs::pacman::{Pacman, PacmanConfig};
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use crate::envs::survival::{Survival, config::SurvivalConfig};
    use crate::make::{EnvConfig, MultiEnvSpec};
    use crate::render::env::GridRenderState;
    use crate::timestep::TimeStepBuffers;
    use crate::wrappers::multitask::MultitaskWrapper;
    use tileset::{TILESET_COLS, TILESET_ROWS};

    /// One of each env, at its defaults, to check the tables against.
    fn every_env() -> Vec<Box<dyn Environment>> {
        vec![
            Box::new(FindReturn::new(&FindReturnConfig::default(), 512)),
            Box::new(Scouts::new(&ScoutsConfig::default(), 512)),
            Box::new(Pacman::new(&PacmanConfig::default(), 512)),
            Box::new(Survival::new(&SurvivalConfig::default(), 512)),
            Box::new(Maze::new(&MazeConfig::default(), 512)),
            Box::new(HideSeek::new(&HideSeekConfig::default(), 512)),
        ]
    }

    /// Off-grid coordinates would sample a uv rect from past the edge of the
    /// texture, which the sampler happily clamps into something wrong-looking
    /// rather than reporting.
    #[test]
    fn every_art_tile_is_on_the_grid() {
        for (label, col, row) in TILE_ART {
            assert!(
                *col < TILESET_COLS && *row < TILESET_ROWS,
                "{label} off grid"
            );
        }
    }

    /// [`resolve_art`] panics on a symbol without art; catch that here, where
    /// the failure names the symbol, instead of at first launch.
    #[test]
    fn every_env_symbol_has_art() {
        for env in every_env() {
            for symbol in env.obs_vocab().symbols() {
                assert!(
                    TILE_ART.iter().any(|(name, _, _)| name == symbol),
                    "no art for {symbol}"
                );
            }
        }
    }

    /// The art table is indexed by the ids the renderer reads back, which a
    /// multitask wrapper remaps into its union vocab. Reporting the inner
    /// env's vocab instead sizes the table for pre-remap ids: pac-man's own
    /// symbols start past the ones find_return contributes first, which runs
    /// off the end of pac-man's table on the first POV frame.
    #[test]
    fn wrapped_env_render_ids_index_the_settings_vocab() {
        let pacman = EnvConfig::RustPacman(Box::new(PacmanConfig {
            view_width: 15,
            view_height: 15,
            ..Default::default()
        }));
        let find_return = EnvConfig::RustFindReturn(Box::new(FindReturnConfig {
            width: 16,
            height: 16,
            view_width: 15,
            view_height: 15,
            ..Default::default()
        }));
        let specs = [
            MultiEnvSpec {
                name: "find_return".to_owned(),
                num: 1,
                env: Box::new(find_return),
            },
            MultiEnvSpec {
                name: "pacman".to_owned(),
                num: 1,
                env: Box::new(pacman),
            },
        ];

        let mut env = MultitaskWrapper::new(&specs, 64).expect("multi env builds");
        env.set_enjoy_mode(Some(1));

        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());
        let mut render_state = GridRenderState::default();
        env.render_state_into(&mut render_state);

        let settings = env.get_render_settings();
        let art = resolve_art(&settings.obs_vocab);
        for &id in buffers.obs.iter().chain(render_state.tilemap.iter()) {
            assert!(
                usize::from(id) < art.len(),
                "id {id} does not index the {} art tiles of {:?}",
                art.len(),
                settings.obs_vocab.symbols(),
            );
        }

        // The ids are the union ones, not the inner env's: pac-man on the
        // focused tile draws as pac-man, and sees itself.
        let tile = render_state.tilemap[render_state.agent_positions[0].idx()];
        let symbol = &settings.obs_vocab.symbols()[usize::from(tile)];
        assert!(
            symbol.starts_with("agent/pacman_"),
            "focused tile is {symbol}"
        );
        assert!(
            buffers.obs.iter().any(|&id| id == tile),
            "pac-man is in its own observation"
        );
    }
}
