//! The callback side of the demo control model: the render loop owns the
//! stepping and asks a [`Policy`] for every agent's action once per step.
//! Training keeps the inverse model (the agent owns the loop) through
//! [`Environment`](crate::env::Environment) directly.

use ndarray::{ArrayView1, ArrayView2, ArrayView4};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Read-only views of the current timestep, borrowed from whoever owns the
/// buffers. `time`/`last_action`/`task_ids` are omitted until a policy needs
/// them.
pub struct PolicyInputs<'a> {
    /// `(num_agents, view_width, view_height, OBS_CHANNELS)`
    pub obs: ArrayView4<'a, u8>,
    pub reward: ArrayView1<'a, f32>,
    pub terminated: ArrayView1<'a, bool>,
    /// `(num_agents, num_actions)`, true marks a legal action.
    pub action_mask: ArrayView2<'a, bool>,
}

pub type PolicyError = Box<dyn std::error::Error + Send + Sync>;

/// Called once per env step to fill in one action id per agent. `Send`
/// because the boxed policy crosses `py.detach` in the python bindings.
pub trait Policy: Send {
    fn act(&mut self, inputs: &PolicyInputs<'_>, actions: &mut [i32]) -> Result<(), PolicyError>;
}

/// Uniform over each agent's legal actions.
pub struct RandomPolicy {
    rng: StdRng,
    /// Scratch for the legal-action ids of one agent, reused across calls.
    legal: Vec<i32>,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            legal: Vec::new(),
        }
    }
}

impl Policy for RandomPolicy {
    fn act(&mut self, inputs: &PolicyInputs<'_>, actions: &mut [i32]) -> Result<(), PolicyError> {
        let num_actions = inputs.action_mask.ncols();
        for (agent, action) in actions.iter_mut().enumerate() {
            self.legal.clear();
            for (id, &legal) in inputs.action_mask.row(agent).iter().enumerate() {
                if legal {
                    self.legal.push(id as i32);
                }
            }
            // an env that emits an all-false row gets uniform over everything
            // rather than a panic; no such env exists today
            *action = if self.legal.is_empty() {
                self.rng.random_range(0..num_actions as i32)
            } else {
                self.legal[self.rng.random_range(0..self.legal.len())]
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    fn act_with_mask(mask: Array2<bool>) -> Vec<i32> {
        let num_agents = mask.nrows();
        let obs = Array4::zeros((num_agents, 1, 1, 1));
        let reward = Array1::zeros(num_agents);
        let terminated = Array1::default(num_agents);
        let inputs = PolicyInputs {
            obs: obs.view(),
            reward: reward.view(),
            terminated: terminated.view(),
            action_mask: mask.view(),
        };

        let mut actions = vec![-1; num_agents];
        RandomPolicy::new(0)
            .act(&inputs, &mut actions)
            .expect("random policy is infallible");
        actions
    }

    #[test]
    fn samples_only_legal_actions() {
        // each agent's only legal action is its own index
        let mask = Array2::from_shape_fn((8, 8), |(agent, action)| agent == action);
        let actions = act_with_mask(mask);
        assert_eq!(actions, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn all_false_mask_falls_back_to_any_action() {
        let actions = act_with_mask(Array2::default((16, 4)));
        assert!(actions.iter().all(|&a| (0..4).contains(&a)));
    }
}
