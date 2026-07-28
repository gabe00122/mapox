use rand::rngs::SmallRng;

pub trait Env {
    type EnvState;

    fn create(self, rngs: &mut SmallRng) -> Self::EnvState;
    fn reset(self, state: &mut Self::EnvState, rngs: &mut SmallRng);
}
