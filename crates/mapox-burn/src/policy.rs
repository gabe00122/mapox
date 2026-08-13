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
        if self.carry.time > 0 {
            log::info!(
                "[{} steps] mean reward per agent: {:.3}",
                self.carry.time,
                self.episode_reward / self.num_agents as f64
            );
        }
        self.episode_reward = 0.0;
        self.rng = SmallRng::seed_from_u64(seed);

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
