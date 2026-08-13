//! [`BurnPolicy`] adapts the model to mapox-core's [`Policy`] callback, the
//! rust twin of `RunPolicy` in jaxrl's `scripts/demo_policy.py`. Everything the
//! model conditions on comes off the timestep, `last_action` included: the env
//! writes the action it actually ran, so a keyboard override is reflected
//! rather than missed. The only state left here is the run's rng and its
//! reward tally — the decode cursor lives in [`Carry`], next to the cache it
//! indexes.
//!
//! The model is in distribution for `max_seq_length` consecutive steps at most:
//! the kv cache and the rope tables are both that long, and the model trained
//! on episodes that long. That is a property of the export, not a setting —
//! a shorter context is a different model — so a driver reads it off
//! [`BurnPolicy::context_length`] and resets at least that often.
//! [`Policy::act`] reports a driver that does not as an error rather than
//! silently restarting the model's context underneath a running episode.
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
    /// Samples the action distribution (with our own rng, not jax's).
    rng: SmallRng,
    episode_reward: f64,
}

impl<B: Backend> BurnPolicy<B> {
    pub fn new(loaded: LoadedPolicy<B>, num_agents: usize, seed: u64) -> Self {
        let model = loaded.model;
        Self {
            carry: model.init_carry(num_agents),
            num_agents,
            rng: SmallRng::seed_from_u64(seed),
            episode_reward: 0.0,
            model,
        }
    }

    /// The longest run of steps this policy stays in distribution for, and so
    /// the episode length a driver should run it at: that makes the env reset
    /// and the end of the model's context the same event.
    ///
    /// Read off the export rather than configurable — the cache and the rope
    /// tables are built to this length and the model trained at it, so a
    /// shorter context is a different model, not a setting.
    pub fn context_length(&self) -> usize {
        self.model.max_seq_length
    }
}

impl<B: Backend> Policy for BurnPolicy<B> {
    fn act(
        &mut self,
        timestep: &TimeStepRef<'_>,
        actions: &mut [VocabId],
    ) -> Result<(), PolicyError> {
        if self.carry.time >= self.context_length() {
            return Err(format!(
                "{} steps without a reset: a driver must reset this policy at least every \
                 context_length ({}) steps",
                self.carry.time,
                self.context_length()
            )
            .into());
        }
        self.episode_reward += timestep.reward.iter().map(|&r| r as f64).sum::<f64>();

        log::info!("{} step", self.carry.time);

        let reward: Vec<f32> = timestep.reward.iter().copied().collect();
        let last_action = timestep
            .last_action
            .as_slice()
            .ok_or("the timestep's last_action view is not contiguous")?;
        let log_probs = self.model.step(
            timestep.obs,
            &reward,
            last_action,
            timestep.action_mask,
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

        Ok(())
    }

    fn reset(&mut self, num_agents: usize, seed: u64) -> Result<(), PolicyError> {
        // before the `num_agents` update below, which the mean divides by.
        // `log` rather than `println!`: on wasm stdout goes nowhere, and the
        // browser build routes this to the console through eframe's WebLogger
        if self.carry.time > 0 {
            log::info!(
                "[{} steps] mean reward per agent: {:.3}",
                self.carry.time,
                self.episode_reward / self.num_agents as f64
            );
        }
        self.episode_reward = 0.0;
        self.rng = SmallRng::seed_from_u64(seed);

        // the viewer's env is fixed for the life of the app, so the resize only
        // guards the contract; the cache is shaped [batch, ..] and cannot be
        // reused across a batch change
        if num_agents == self.num_agents {
            self.carry.rewind();
        } else {
            self.num_agents = num_agents;
            self.carry = self.model.init_carry(num_agents);
        }
        Ok(())
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
