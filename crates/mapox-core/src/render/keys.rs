use egui::{InputState, Key};

use crate::{
    symbols,
    vocab::{VocabId, Vocabulary},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    ToggleView,
    ToggleFov,
    TogglePacing,
    NextAgent,
    NextTask,
    ToggleSidePanel,
    Reset,
    ShowControls,
    Quit,
}

impl Command {
    fn available(self) -> bool {
        !(matches!(self, Self::Quit) && cfg!(target_arch = "wasm32"))
    }

    /// What the command does, as the key reference lists it.
    pub fn label(self) -> &'static str {
        match self {
            Self::ToggleView => "bird's-eye / agent view",
            Self::ToggleFov => "field of view on / off",
            Self::TogglePacing => "free-run / step-on-input",
            Self::NextAgent => "next agent",
            Self::NextTask => "next task",
            Self::ToggleSidePanel => "side panel open / closed",
            Self::Reset => "reset episode",
            Self::ShowControls => "this list",
            Self::Quit => "quit",
        }
    }
}

const COMMAND_KEYS: &[(Key, Command)] = &[
    (Key::C, Command::ToggleView),
    (Key::V, Command::ToggleFov),
    (Key::P, Command::TogglePacing),
    (Key::N, Command::NextAgent),
    (Key::T, Command::NextTask),
    (Key::H, Command::ToggleSidePanel),
    (Key::R, Command::Reset),
    (Key::Questionmark, Command::ShowControls),
    (Key::Escape, Command::Quit),
];

/// Each action symbol with the name a person reads and the keys that send it.
const ACTION_KEYS: &[(&str, &str, &[Key])] = &[
    (symbols::MOVE_UP, "move up", &[Key::ArrowUp, Key::W]),
    (
        symbols::MOVE_RIGHT,
        "move right",
        &[Key::ArrowRight, Key::D],
    ),
    (symbols::MOVE_DOWN, "move down", &[Key::ArrowDown, Key::S]),
    (symbols::MOVE_LEFT, "move left", &[Key::ArrowLeft, Key::A]),
    (symbols::NOOP, "wait", &[Key::Period]),
    (symbols::PRIMARY_ACTION, "primary action", &[Key::F]),
    (symbols::DIG_ACTION, "dig", &[Key::E]),
    (symbols::PLACE_PIPE, "place pipe", &[Key::Q]),
    (symbols::TAKE, "take", &[Key::E]),
    (symbols::PUT, "put", &[Key::Q]),
    (symbols::SWAP1, "swap2", &[Key::Num2]),
    (symbols::SWAP2, "swap3", &[Key::Num3]),
    (symbols::SWAP3, "swap4", &[Key::Num4]),
    (symbols::SWAP4, "swap5", &[Key::Num5]),
    (symbols::COMBINE, "craft", &[Key::X]),
    (symbols::USE, "use", &[Key::F]),
    (symbols::REFINE, "refine", &[Key::G]),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    Command(Command),
    Action(VocabId),
}

/// The commands this build responds to, with their keys, in display order.
pub fn command_bindings() -> impl Iterator<Item = (Key, Command)> {
    COMMAND_KEYS
        .iter()
        .copied()
        .filter(|(_, command)| command.available())
}

/// The actions in `action_vocab` that a key sends, in display order: each
/// action's id, label, and the keys bound to it. Bindings for symbols the
/// vocab lacks are left out, as [`read`] ignores their keys.
pub fn action_bindings(
    action_vocab: &Vocabulary,
) -> impl Iterator<Item = (VocabId, &'static str, &'static [Key])> + '_ {
    ACTION_KEYS
        .iter()
        .filter_map(|&(symbol, label, keys)| action_vocab.get(symbol).map(|id| (id, label, keys)))
}

/// How the viewer names the action `symbol`; one it has no key for keeps its
/// symbol.
pub fn action_label(symbol: &str) -> &str {
    ACTION_KEYS
        .iter()
        .find(|&&(bound, _, _)| bound == symbol)
        .map_or(symbol, |&(_, label, _)| label)
}

/// How the key reference prints `key`: arrows as glyphs, the rest by name.
pub fn key_label(key: Key) -> &'static str {
    match key {
        Key::Escape => "Esc",
        key => key.symbol_or_name(),
    }
}

pub fn read(state: &InputState, action_vocab: &Vocabulary) -> Option<Input> {
    let command = command_bindings()
        .find(|&(key, _)| state.key_pressed(key))
        .map(|(_, command)| Input::Command(command));

    command.or_else(|| {
        action_bindings(action_vocab)
            .find(|(_, _, keys)| keys.iter().any(|&key| state.key_pressed(key)))
            .map(|(id, _, _)| Input::Action(id))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use crate::envs::pacman::{Pacman, PacmanConfig};
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use crate::envs::survival::{Survival, config::SurvivalConfig};
    use egui::{Event, Modifiers};

    fn all_envs() -> Vec<Box<dyn Environment>> {
        vec![
            Box::new(FindReturn::new(&FindReturnConfig::default(), 512)),
            Box::new(Scouts::new(&ScoutsConfig::default(), 512)),
            Box::new(Pacman::new(&PacmanConfig::default(), 512)),
            Box::new(Survival::new(&SurvivalConfig::default(), 512)),
        ]
    }

    /// One frame's worth of input with `keys` freshly pressed. Assigned
    /// rather than struct-literalled: `InputState` keeps some fields private.
    fn frame(keys: &[Key]) -> InputState {
        let mut state = InputState::default();
        state.events = keys
            .iter()
            .map(|&key| Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            })
            .collect();
        state
    }

    /// An action no key sends is an action the env can't be play-tested on.
    #[test]
    fn every_env_action_has_a_key() {
        for env in all_envs() {
            for symbol in env.action_vocab().symbols() {
                assert!(
                    ACTION_KEYS.iter().any(|(name, _, _)| name == symbol),
                    "no key bound to {symbol}"
                );
            }
        }
    }

    /// The panel shows the last action by label, and an action the viewer
    /// has no row for still shows up, under its symbol.
    #[test]
    fn actions_are_labelled_for_people() {
        assert_eq!(action_label(symbols::PLACE_PIPE), "place pipe");
        assert_eq!(action_label("teleport"), "teleport");
    }

    /// A command key means the same thing whatever env is loaded, so it must
    /// not also send an action.
    #[test]
    fn command_keys_are_unique_and_never_actions() {
        let mut seen = Vec::new();
        for &(key, _) in COMMAND_KEYS {
            assert!(!seen.contains(&key), "{} is bound twice", key.name());
            seen.push(key);
        }

        for (symbol, _, keys) in ACTION_KEYS {
            for key in *keys {
                assert!(
                    !seen.contains(key),
                    "{} is both a command and {symbol}",
                    key.name()
                );
            }
        }
    }

    /// Action keys may be shared between envs that never have both actions
    /// (E is dig in one, take in another), but within an env a key must send
    /// one action: [`read`] would silently take the earlier row.
    #[test]
    fn no_key_sends_two_actions_in_one_env() {
        for env in all_envs() {
            let mut seen = Vec::new();
            for (_, label, keys) in action_bindings(env.action_vocab()) {
                for &key in keys {
                    assert!(
                        !seen.contains(&key),
                        "{} sends two actions, one of them {label}",
                        key.name()
                    );
                    seen.push(key);
                }
            }
        }
    }

    /// The whole point of binding by symbol: a key an env has no action for
    /// does nothing, rather than sending whatever id sits at that position.
    #[test]
    fn keys_the_env_has_no_action_for_are_inert() {
        let vocab: Vocabulary = [symbols::MOVE_UP, symbols::DIG_ACTION]
            .into_iter()
            .collect();

        assert_eq!(
            read(&frame(&[Key::E]), &vocab),
            Some(Input::Action(vocab.get(symbols::DIG_ACTION).unwrap()))
        );
        // Q is bound to place_pipe, which this vocab doesn't have
        assert_eq!(read(&frame(&[Key::Q]), &vocab), None);
        assert_eq!(read(&frame(&[]), &vocab), None);
    }

    /// Both halves of the keyboard in one frame resolve to the command; the
    /// action is dropped rather than applied alongside it.
    #[test]
    fn a_command_beats_an_action_in_the_same_frame() {
        let vocab = FindReturn::new(&FindReturnConfig::default(), 512)
            .action_vocab()
            .clone();

        assert_eq!(
            read(&frame(&[Key::W, Key::R]), &vocab),
            Some(Input::Command(Command::Reset))
        );
    }
}
