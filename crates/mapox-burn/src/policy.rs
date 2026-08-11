//! [`BurnPolicy`] adapts the model to mapox-core's [`Policy`] callback, the
//! rust twin of `RunPolicy` in jaxrl's `scripts/demo_policy.py`: the viewer
//! only hands over `(obs, reward, terminated, action_mask)`, so the two other
//! inputs the model conditions on — `time` and `last_action` — are tracked
//! here, and the carry is cycled every `memory_steps` calls because the kv
//! cache holds `max_seq_length` entries and the model was trained on episodes
//! that long. The same caveats apply: a keyboard override makes one agent's
//! `last_action` wrong for a step, and an env reset leaves the tracked state
//! stale until the next cycle.
//!
//! The backend is a CPU one on both targets, so the logits are already in host
//! memory and [`Policy::act`] fills in the actions in one call — natively and
//! in the browser alike. A device backend would need an async readback, which
//! neither this trait nor the render loop has a shape for.

use burn::tensor::backend::Backend;
use mapox_core::policy::{Policy, PolicyError};
use mapox_core::timestep::TimeStepRef;
use mapox_core::vocab::VocabId;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::loader::LoadedPolicy;
use crate::model::{Carry, TransformerActor};

pub struct BurnPolicy<B: Backend> {
    model: TransformerActor<B>,
    carry: Carry<B>,
    num_agents: usize,
    memory_steps: usize,
    time: usize,
    last_action: Vec<u16>,
    /// Samples the action distribution (with our own rng, not jax's).
    rng: SmallRng,
    episode_reward: f64,
}

impl<B: Backend> BurnPolicy<B> {
    pub fn new(
        loaded: LoadedPolicy<B>,
        num_agents: usize,
        memory_steps: Option<usize>,
        seed: u64,
    ) -> Self {
        let model = loaded.model;
        Self {
            carry: model.init_carry(num_agents),
            num_agents,
            // capped, not just defaulted: the cache and the rope tables are
            // both `max_seq_length` long, and running past them is out of
            // distribution anyway (the model trained on episodes that long).
            memory_steps: memory_steps
                .unwrap_or(model.max_seq_length)
                .clamp(1, model.max_seq_length),
            time: 0,
            last_action: vec![0; num_agents],
            rng: SmallRng::seed_from_u64(seed),
            episode_reward: 0.0,
            model,
        }
    }

    fn cycle_memory(&mut self) {
        // `log` rather than `println!`: on wasm stdout goes nowhere, and the
        // browser build routes this to the console through eframe's WebLogger.
        log::info!(
            "[{} steps] mean reward per agent: {:.3}",
            self.memory_steps,
            self.episode_reward / self.num_agents as f64
        );
        self.time = 0;
        self.episode_reward = 0.0;
        self.last_action.fill(0);
    }
}

impl<B: Backend> Policy for BurnPolicy<B> {
    fn act(
        &mut self,
        timestep: &TimeStepRef<'_>,
        actions: &mut [VocabId],
    ) -> Result<(), PolicyError> {
        if self.time >= self.memory_steps {
            self.cycle_memory();
        }
        self.episode_reward += timestep.reward.iter().map(|&r| r as f64).sum::<f64>();

        let reward: Vec<f32> = timestep.reward.iter().copied().collect();
        let log_probs = self.model.step(
            timestep.obs,
            &reward,
            &self.last_action,
            timestep.action_mask,
            self.time,
            &mut self.carry,
        );
        let num_actions = log_probs.dims()[1];
        let log_probs = log_probs
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| format!("reading log probs: {e:?}"))?;

        for (agent, action) in actions.iter_mut().enumerate() {
            let row = &log_probs[agent * num_actions..(agent + 1) * num_actions];
            *action = VocabId::try_from(sample(row, &mut self.rng))
                .expect("action index exceeds VocabId");
        }

        // This will be wrong if the user overrides the agents action with a keyboard input, it should be looped through the environment not remembered
        self.last_action.copy_from_slice(actions);
        self.time += 1;
        Ok(())
    }

    fn reset(&mut self) {
        self.cycle_memory();
    }
}

fn sample(log_probs: &[f32], rng: &mut SmallRng) -> usize {
    let mut r: f32 = rng.random();
    for (i, &lp) in log_probs.iter().enumerate() {
        r -= lp.exp();
        if r <= 0.0 {
            return i;
        }
    }
    // float round-off leftovers land on the last action
    log_probs.len() - 1
}
