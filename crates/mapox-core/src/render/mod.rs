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
/// is terrain, row 1 items and characters, row 2 the snake colours.
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
    (symbols::TILE_FOOD, 2, 1),
    (symbols::AGENT_GENERIC, 7, 1),
    (symbols::AGENT_SCOUT, 8, 1),
    (symbols::AGENT_HARVESTER, 9, 1),
    (symbols::AGENT_SNAKE_RED, 0, 2),
    (symbols::AGENT_SNAKE_ORANGE, 1, 2),
    (symbols::AGENT_SNAKE_YELLOW, 2, 2),
    (symbols::AGENT_SNAKE_GOLD, 3, 2),
    (symbols::AGENT_SNAKE_GREEN, 4, 2),
    (symbols::AGENT_SNAKE_BLUE, 5, 2),
    (symbols::AGENT_SNAKE_PURPLE, 6, 2),
    (symbols::AGENT_SNAKE_PINK, 7, 2),
    (symbols::AGENT_SNAKE_GRAY, 8, 2),
    (symbols::AGENT_SNAKE_WHITE, 9, 2),
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
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use crate::envs::snake::{Snake, SnakeConfig};
    use crate::make::{EnvConfig, MultiEnvSpec};
    use crate::render::env::GridRenderState;
    use crate::symbols::AGENT_SNAKE_RED;
    use crate::timestep::TimeStepBuffers;
    use crate::wrappers::multitask::MultitaskWrapper;
    use tileset::{TILESET_COLS, TILESET_ROWS};

    /// One of each env, at its defaults, to check the tables against.
    fn every_env() -> Vec<Box<dyn Environment>> {
        vec![
            Box::new(FindReturn::new(&FindReturnConfig::default(), 512)),
            Box::new(Scouts::new(&ScoutsConfig::default(), 512)),
            Box::new(Snake::new(&SnakeConfig::default(), 512)),
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
    /// env's vocab instead sizes the table for pre-remap ids: snake's colours
    /// start at union id 16 (find_return contributes 15 symbols first), which
    /// runs off the end of snake's 14-entry table on the first POV frame.
    #[test]
    fn wrapped_env_render_ids_index_the_settings_vocab() {
        let snake = EnvConfig::RustSnake(Box::new(SnakeConfig {
            num_agents: 1,
            width: 16,
            height: 16,
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
                name: "snake".to_owned(),
                num: 1,
                env: Box::new(snake),
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

        // The ids are the union ones, not the inner env's: the snake sitting
        // on the focused tile draws as a red snake, and sees its own head.
        let tile = render_state.tilemap[render_state.agent_positions[0].idx()];
        assert_eq!(
            settings.obs_vocab.symbols()[usize::from(tile)],
            AGENT_SNAKE_RED
        );
        assert!(
            buffers
                .obs
                .iter()
                .any(|&id| id == settings.obs_vocab.get(AGENT_SNAKE_RED).unwrap()),
            "the agent's own head is in its observation"
        );
    }
}
