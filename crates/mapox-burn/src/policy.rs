use burn::tensor::backend::Backend;
use mapox_core::policy::{Policy, PolicyError};
use mapox_core::timestep::TimeStepRef;
use mapox_core::vocab::VocabId;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::loader::LoadedPolicy;
use crate::model::{Carry, TransformerActor};

pub struct BurnPolicy<B: Backend> {
    model: TransformerActor<B>,
    /// Allocated by [`Policy::reset`], sized for the agents the driver will
    /// actually step: a multitask env only settles its agent count once the
    /// driver picks an enjoy-mode task, and a cache sized for the whole
    /// training batch can run to gigabytes.
    carry: Option<Carry<B>>,
    num_agents: usize,
    /// Samples the action distribution (with our own rng, not jax's).
    rng: SmallRng,
    episode_reward: f64,
}

impl<B: Backend> BurnPolicy<B> {
    /// The kv cache is not allocated until the first [`Policy::reset`].
    pub fn new(loaded: LoadedPolicy<B>, seed: u64) -> Self {
        Self {
            model: loaded.model,
            carry: None,
            num_agents: 0,
            rng: SmallRng::seed_from_u64(seed),
            episode_reward: 0.0,
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
        let context_length = self.context_length();
        let carry = self
            .carry
            .as_mut()
            .ok_or("act before reset: a driver must reset this policy first")?;
        if carry.time >= context_length {
            return Err(format!(
                "{} steps without a reset: a driver must reset this policy at least every \
                 context_length ({context_length}) steps",
                carry.time,
            )
            .into());
        }
        self.episode_reward += timestep.reward.iter().map(|&r| r as f64).sum::<f64>();

        let reward: Vec<f32> = timestep.reward.iter().copied().collect();
        let last_action = timestep
            .last_action
            .as_slice()
            .ok_or("the timestep's last_action view is not contiguous")?;
        let task_ids = timestep
            .task_ids
            .as_slice()
            .ok_or("the timestep's task_ids view is not contiguous")?;
        let log_probs = self.model.step(
            timestep.obs,
            &reward,
            last_action,
            task_ids,
            timestep.action_mask,
            carry,
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
        if let Some(carry) = &self.carry
            && carry.time > 0
        {
            log::info!(
                "[{} steps] mean reward per agent: {:.3}",
                carry.time,
                self.episode_reward / self.num_agents as f64
            );
        }
        self.episode_reward = 0.0;
        self.rng = SmallRng::seed_from_u64(seed);

        match &mut self.carry {
            Some(carry) if num_agents == self.num_agents => carry.rewind(),
            _ => {
                // drop the old cache before allocating its replacement
                self.carry = None;
                self.num_agents = num_agents;
                self.carry = Some(self.model.init_carry(num_agents));
            }
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
