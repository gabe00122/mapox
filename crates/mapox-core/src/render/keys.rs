use egui::{InputState, Key};

use crate::{
    symbols,
    vocab::{VocabId, Vocabulary},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    ToggleView,
    TogglePacing,
    NextAgent,
    Reset,
    Quit,
}

impl Command {
    fn available(self) -> bool {
        !(matches!(self, Self::Quit) && cfg!(target_arch = "wasm32"))
    }
}

const COMMAND_KEYS: &[(Key, Command)] = &[
    (Key::Tab, Command::ToggleView),
    (Key::P, Command::TogglePacing),
    (Key::N, Command::NextAgent),
    (Key::R, Command::Reset),
    (Key::Escape, Command::Quit),
];

const ACTION_KEYS: &[(&str, &[Key])] = &[
    (symbols::MOVE_UP, &[Key::ArrowUp, Key::W]),
    (symbols::MOVE_RIGHT, &[Key::ArrowRight, Key::D]),
    (symbols::MOVE_DOWN, &[Key::ArrowDown, Key::S]),
    (symbols::MOVE_LEFT, &[Key::ArrowLeft, Key::A]),
    (symbols::STAY, &[Key::Space]),
    (symbols::NOOP, &[Key::Period]),
    (symbols::PRIMARY_ACTION, &[Key::F]),
    (symbols::DIG_ACTION, &[Key::E]),
    (symbols::PLACE_PIPE, &[Key::Q]),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    Command(Command),
    Action(VocabId),
}

pub fn read(state: &InputState, action_vocab: &Vocabulary) -> Option<Input> {
    let command = COMMAND_KEYS
        .iter()
        .filter(|(_, command)| command.available())
        .find(|(key, _)| state.key_pressed(*key))
        .map(|&(_, command)| Input::Command(command));

    command.or_else(|| {
        ACTION_KEYS
            .iter()
            .find(|(_, keys)| keys.iter().any(|key| state.key_pressed(*key)))
            .and_then(|(symbol, _)| action_vocab.get(symbol))
            .map(Input::Action)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use crate::envs::pacman::{Pacman, PacmanConfig};
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use egui::{Event, Modifiers};

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
        let envs: Vec<Box<dyn Environment>> = vec![
            Box::new(FindReturn::new(&FindReturnConfig::default(), 512)),
            Box::new(Scouts::new(&ScoutsConfig::default(), 512)),
            Box::new(Pacman::new(&PacmanConfig::default(), 512)),
        ];

        for env in envs {
            for symbol in env.action_vocab().symbols() {
                assert!(
                    ACTION_KEYS.iter().any(|(name, _)| name == symbol),
                    "no key bound to {symbol}"
                );
            }
        }
    }

    /// Deliberately strict: a key means one thing in the viewer, whatever env
    /// is loaded, so two rows claiming it is a mistake rather than a fallback.
    /// [`read`] would silently take the earlier row.
    #[test]
    fn no_key_is_bound_twice() {
        let mut bound: Vec<Key> = COMMAND_KEYS.iter().map(|&(key, _)| key).collect();
        bound.extend(
            ACTION_KEYS
                .iter()
                .flat_map(|(_, keys)| keys.iter().copied()),
        );

        let mut seen = Vec::new();
        for key in bound {
            assert!(!seen.contains(&key), "{} is bound twice", key.name());
            seen.push(key);
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
