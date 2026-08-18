use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::timestep::TimeStepRef;
use crate::vocab::VocabId;

pub type PolicyError = Box<dyn std::error::Error + Send + Sync>;

pub trait Policy: Send {
    fn act(
        &mut self,
        timestep: &TimeStepRef<'_>,
        actions: &mut [VocabId],
    ) -> Result<(), PolicyError>;

    fn reset(&mut self, num_agents: usize, seed: u64) -> Result<(), PolicyError>;
}

pub struct RandomPolicy {
    rng: SmallRng,
    legal: Vec<VocabId>,
}

impl RandomPolicy {
    pub fn new() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(0),
            legal: Vec::new(),
        }
    }
}

impl Default for RandomPolicy {
    fn default() -> Self {
        RandomPolicy::new()
    }
}

impl Policy for RandomPolicy {
    fn act(
        &mut self,
        timestep: &TimeStepRef<'_>,
        actions: &mut [VocabId],
    ) -> Result<(), PolicyError> {
        // TODO: Surely this could be simpler

        for (agent, action) in actions.iter_mut().enumerate() {
            self.legal.clear();
            for (id, &legal) in timestep.action_mask.row(agent).iter().enumerate() {
                if legal {
                    self.legal
                        .push(VocabId::try_from(id).expect("action mask exceeds VocabId capacity"));
                }
            }
            // an env that emits an all-false row gets uniform over everything
            // rather than a panic; no such env exists today
            *action = self.legal[self.rng.random_range(0..self.legal.len())];
        }
        Ok(())
    }

    fn reset(&mut self, _num_agents: usize, seed: u64) -> Result<(), PolicyError> {
        self.rng = SmallRng::seed_from_u64(seed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestep::TimeStepBuffers;
    use ndarray::Array2;

    fn act_with_mask(mask: Array2<bool>) -> Vec<VocabId> {
        let num_agents = mask.nrows();
        let mut buffers = TimeStepBuffers::with_shape(num_agents, 1, 1, mask.ncols());
        buffers.action_mask.assign(&mask);

        let mut actions = vec![VocabId::MAX; num_agents];
        RandomPolicy::new()
            .act(&buffers.view(), &mut actions)
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
