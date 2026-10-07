//! What agents can do, and doing it. Every action works on the tile in front
//! of the agent; the action mask is [`Survival::can`], so what a policy may
//! pick and what `act` carries out never disagree.

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
    items::{Item, Job, Work, recipe},
    metrics::Achievement,
    survivor::Survivor,
    tiles::SurvivalObs,
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

impl Survival {
    fn can(&self, agent: &Survivor, action: SurvivalAction) -> bool {
        use SurvivalAction::*;
        if agent.work.is_some() {
            return action == Noop;
        }
        let ahead = self.state.map[agent.ahead().idx()];
        match action {
            MoveUp | MoveRight | MoveDown | MoveLeft | Noop => true,
            Grab => {
                (agent.hands.is_none() || agent.backpack.is_none())
                    && (ahead.item().is_some()
                        || matches!(
                            ahead,
                            SurvivalObs::TileBerryBush | SurvivalObs::TileBuriedCarrot
                        ))
            }
            Drop => agent.hands.is_some() && ahead.is_floor(),
            Swap => agent.hands.is_some() || agent.backpack.is_some(),
            Use => match agent.hands {
                Some(Item::Wood) => ahead == SurvivalObs::TileFireLow,
                Some(Item::Campfire) => ahead.is_floor(),
                Some(food) if food.cooked().is_some() => ahead.is_fire(),
                // a tool's job, or with empty hands one done by hand
                tool => Job::of(tool, ahead).is_some(),
            },
            Eat => agent
                .hands
                .is_some_and(|item| item.food(&self.config).is_some()),
            Combine => match (agent.hands, agent.backpack) {
                (Some(a), Some(b)) => recipe(a, b).is_some(),
                _ => false,
            },
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
        if !self.can(&agent, action) {
            return;
        }

        let ahead = agent.ahead();
        match action {
            SurvivalAction::MoveUp
            | SurvivalAction::MoveRight
            | SurvivalAction::MoveDown
            | SurvivalAction::MoveLeft => {
                agent.dir = action.heading().expect("moves have a heading");
                let target = agent.ahead();
                if self.state.map[target.idx()].walkable() {
                    self.remove_creature(agent.position);
                    agent.position = target;
                }
                self.place_creature(agent.position, agent.tile());
            }
            SurvivalAction::Grab => {
                let item = match self.state.map[ahead.idx()] {
                    SurvivalObs::TileBerryBush => {
                        self.pick_bush(ahead);
                        Item::Berry
                    }
                    SurvivalObs::TileBuriedCarrot => {
                        self.set_ground(ahead, SurvivalObs::TileEmpty);
                        Item::Carrot
                    }
                    _ => self.take(ahead),
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
            SurvivalAction::Drop => {
                let item = agent
                    .hands
                    .take()
                    .expect("drop is legal only holding something");
                self.lay(ahead, item);
            }
            SurvivalAction::Swap => std::mem::swap(&mut agent.hands, &mut agent.backpack),
            SurvivalAction::Use => match agent.hands {
                Some(Item::Wood) => {
                    self.stoke(ahead);
                    agent.hands = None;
                    agent.unlock(&mut self.metrics, Achievement::RefuelFire);
                }
                Some(Item::Campfire) => {
                    self.kindle(ahead);
                    agent.hands = None;
                    agent.unlock(&mut self.metrics, Achievement::PlaceFire);
                }
                Some(food) if food.cooked().is_some() => {
                    let (cooked, achievement) = food.cooked().expect("matched on cooking");
                    agent.hands = Some(cooked);
                    agent.unlock(&mut self.metrics, achievement);
                }
                tool => {
                    let job = Job::of(tool, self.state.map[ahead.idx()])
                        .expect("use is legal only where there is a job to do");
                    self.start(&mut agent, job);
                }
            },
            SurvivalAction::Eat => {
                let (food, achievement) = agent
                    .hands
                    .and_then(|item| item.food(&self.config))
                    .expect("eat is legal only holding food");
                agent.eat(food);
                agent.hands = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            SurvivalAction::Combine => {
                let (hands, backpack) = (agent.hands, agent.backpack);
                let (made, achievement) = hands
                    .zip(backpack)
                    .and_then(|(a, b)| recipe(a, b))
                    .expect("combine is legal only for a recipe");
                agent.hands = Some(made);
                agent.backpack = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            SurvivalAction::Noop => {}
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
                mask[action as usize] = self.can(agent, action);
            }
        }
    }
}
