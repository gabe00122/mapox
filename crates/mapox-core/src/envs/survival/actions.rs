//! What agents can do, and doing it. Every action works on the tile in front
//! of the agent. [`Survival::effect`] works out what an action does, and both
//! the action mask and `act` read it, so what a policy may pick and what
//! `act` carries out never disagree. The rules behind it are the tiles' and
//! items' own ([`SurvivalObs::gather`], [`Usage::of`], [`recipe`], ...).

use crate::{
    symbols::{
        COMBINE_ACTION, DROP_ACTION, EAT_ACTION, GRAB_ACTION, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
        MOVE_UP, NOOP, SWAP_ACTION, USE_ACTION,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

use super::{
    Survival,
    items::{Item, Job, Usage, Work, recipe},
    metrics::Achievement,
    survivor::Survivor,
    tiles::{Gather, SurvivalObs},
};

vocab_enum!(pub(super) SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Grab => GRAB_ACTION,
    Drop => DROP_ACTION,
    Swap => SWAP_ACTION,
    Use => USE_ACTION,
    Eat => EAT_ACTION,
    Combine => COMBINE_ACTION,
    Noop => NOOP,
});

impl SurvivalAction {
    /// The facing a move turns to, as a `DIRECTIONS` index.
    fn heading(self) -> Option<u8> {
        use SurvivalAction::*;
        match self {
            MoveUp => Some(0),
            MoveRight => Some(1),
            MoveDown => Some(2),
            MoveLeft => Some(3),
            _ => None,
        }
    }
}

/// What an action comes to for an agent, worked out in full before any of
/// it is carried out.
#[derive(Debug, Clone, Copy)]
enum Effect {
    /// Turn to the `DIRECTIONS` index, stepping forward if the way is clear.
    Move(u8),
    Gather(Gather),
    Drop(Item),
    Swap,
    Use(Usage),
    Eat(u16, Achievement),
    Combine(Item, Achievement),
    Wait,
}

impl Survival {
    /// What `action` does for `agent` right now, or `None` if it is not
    /// legal: the mask and [`Survival::act`] both read it, so they agree.
    fn effect(&self, agent: &Survivor, action: SurvivalAction) -> Option<Effect> {
        use SurvivalAction::*;
        if agent.work.is_some() {
            // a busy agent can only wait its job out
            return (action == Noop).then_some(Effect::Wait);
        }
        let ahead = self.state.map[agent.ahead().idx()];
        match action {
            MoveUp | MoveRight | MoveDown | MoveLeft => action.heading().map(Effect::Move),
            Grab => {
                let room = agent.hands.is_none() || agent.backpack.is_none();
                ahead.gather().filter(|_| room).map(Effect::Gather)
            }
            Drop => agent.hands.filter(|_| ahead.is_floor()).map(Effect::Drop),
            Swap => (agent.hands.is_some() || agent.backpack.is_some()).then_some(Effect::Swap),
            Use => Usage::of(agent.hands, ahead).map(Effect::Use),
            Eat => agent
                .hands
                .and_then(|item| item.food(&self.config))
                .map(|(food, achievement)| Effect::Eat(food, achievement)),
            Combine => agent
                .hands
                .zip(agent.backpack)
                .and_then(|(a, b)| recipe(a, b))
                .map(|(made, achievement)| Effect::Combine(made, achievement)),
            Noop => Some(Effect::Wait),
        }
    }

    /// Carries out one agent's action, or nothing if the action is not legal
    /// for it right now: an agent earlier in the turn order may have taken
    /// what it reached for.
    pub(super) fn act(&mut self, agent_id: usize, action: SurvivalAction) {
        let mut agent = self.state.agents[agent_id];
        if agent.work.is_some() {
            // a busy agent's turn goes into its job, whatever it asked for
            self.work(&mut agent);
            self.state.agents[agent_id] = agent;
            return;
        }
        let Some(effect) = self.effect(&agent, action) else {
            return;
        };

        let ahead = agent.ahead();
        match effect {
            Effect::Move(dir) => {
                agent.dir = dir;
                let target = agent.ahead();
                if self.state.map[target.idx()].walkable() {
                    self.remove_creature(agent.position);
                    agent.position = target;
                }
                self.place_creature(agent.position, agent.tile());
            }
            Effect::Gather(gather) => {
                let item = match gather {
                    Gather::PickBerry => {
                        self.pick_bush(ahead);
                        Item::Berry
                    }
                    Gather::PullCarrot => {
                        self.set_ground(ahead, SurvivalObs::TileEmpty);
                        Item::Carrot
                    }
                    Gather::TakeUp => self.take(ahead),
                };
                // full hands spill over into an empty backpack
                let slot = if agent.hands.is_none() {
                    &mut agent.hands
                } else {
                    &mut agent.backpack
                };
                *slot = Some(item);
                if let Some(achievement) = item.collected() {
                    agent.unlock(&mut self.metrics, achievement);
                }
            }
            Effect::Drop(item) => {
                agent.hands = None;
                self.lay(ahead, item);
            }
            Effect::Swap => std::mem::swap(&mut agent.hands, &mut agent.backpack),
            Effect::Use(Usage::Stoke) => {
                self.stoke(ahead);
                agent.hands = None;
                agent.unlock(&mut self.metrics, Achievement::RefuelFire);
            }
            Effect::Use(Usage::Kindle) => {
                self.kindle(ahead);
                agent.hands = None;
                agent.unlock(&mut self.metrics, Achievement::PlaceFire);
            }
            Effect::Use(Usage::Cook(cooked, achievement)) => {
                agent.hands = Some(cooked);
                agent.unlock(&mut self.metrics, achievement);
            }
            Effect::Use(Usage::Work(job)) => self.start(&mut agent, job),
            Effect::Eat(food, achievement) => {
                agent.eat(food);
                agent.hands = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            Effect::Combine(made, achievement) => {
                agent.hands = Some(made);
                agent.backpack = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            Effect::Wait => {}
        }

        self.state.agents[agent_id] = agent;
    }

    /// Sets the agent to `job` on the tile in front of it.
    fn start(&mut self, agent: &mut Survivor, job: Job) {
        agent.work = Some(Work {
            job,
            target: agent.ahead(),
            steps_left: job.steps(&self.config),
        });
        // the action that starts the job is its first step
        self.work(agent);
    }

    /// Puts a step into the agent's job, finishing it once its steps run out.
    /// If its tile has changed meanwhile (another agent got there first), the
    /// job comes to nothing.
    fn work(&mut self, agent: &mut Survivor) {
        let Some(mut work) = agent.work else {
            return;
        };
        work.steps_left = work.steps_left.saturating_sub(1);
        if work.steps_left > 0 {
            agent.work = Some(work);
            return;
        }

        agent.work = None;
        if work.job.works(self.state.base_map[work.target.idx()]) {
            self.set_ground(work.target, work.job.leaves());
            agent.unlock(&mut self.metrics, work.job.achievement());
        }
    }

    pub(super) fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut mask = timestep.action_mask.row_mut(agent_id);
            for &action in SurvivalAction::TABLE {
                mask[action as usize] = self.effect(agent, action).is_some();
            }
        }
    }
}
