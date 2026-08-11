//! Interactive rendering: [`RenderApp`] owns the egui loop, steps any
//! [`Environment`](crate::env::Environment) under a
//! [`Policy`](crate::policy::Policy), and draws either the whole map or the
//! focused agent's observations. Natively [`open_window`] blocks until the
//! window closes; on the web `mapox-web`'s wasm-bindgen entry point hosts the
//! same app.

pub mod app;
pub mod env;
mod grid;
pub mod tileset;

#[cfg(not(target_arch = "wasm32"))]
pub use app::open_window;
pub use app::{PacingMode, RenderApp, ViewMode};

use crate::{symbols, vocab::Vocabulary};

/// Sheet coordinates per symbol, lifted from the table the pygame renderer
/// uses (`python/mapox/renderer.py::tilemap`). If the atlas math here drifts
/// from that table, the demo is where it shows up first.
const TILE_ART: &[(&str, u32, u32)] = &[
    (symbols::TILE_EMPTY, 17, 0),
    (symbols::TILE_WALL, 20, 3),
    (symbols::TILE_DESTRUCTIBLE_WALL, 20, 3),
    (symbols::TILE_FLAG, 29, 23),
    (symbols::TILE_DECOR_1, 15, 5),
    (symbols::TILE_DECOR_2, 16, 5),
    (symbols::TILE_DECOR_3, 17, 5),
    (symbols::TILE_DECOR_4, 14, 5),
    (symbols::AGENT_GENERIC, 104, 0),
];

/// Sheet coordinates indexed by obs vocab id; panics on a symbol without
/// art, which the `every_env_symbol_has_art` test catches before launch.
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

/// Keybindings shown in the hint bar; input handling in `app.rs` follows
/// this table. `esc` is filtered out on the web where it can't close a tab.
pub(crate) const KEY_HINTS: &[(&str, &str)] = &[
    ("wasd/arrows", "step + move (step mode)"),
    ("tab", "map/pov"),
    ("p", "free-run/step"),
    ("n", "next agent"),
    ("click", "focus agent"),
    ("r", "reset"),
    ("esc", "quit"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use tileset::{TILESET_COLS, TILESET_ROWS};

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
        let env = FindReturn::new(&FindReturnConfig::default());
        for symbol in env.obs_vocab().symbols() {
            assert!(
                TILE_ART.iter().any(|(name, _, _)| name == symbol),
                "no art for {symbol}"
            );
        }
    }
}
