use crate::envs::survival::Survival;

const DAY_TIME: usize = 100;
const DUSK_TIME: usize = 50;
const NIGHT_TIME: usize = 20;
const DUSK_END: usize = DAY_TIME + DUSK_TIME;
const CYCLE_TIME: usize = DUSK_END + NIGHT_TIME;

impl Survival {
    pub(super) fn tick_time(&mut self) {
        let max_radius = self.config.view_width / 2;
        self.state.dusk = match self.state.time % CYCLE_TIME {
            ..DAY_TIME => None,
            // linearly shrink the vision radius from max_radius to 0 over dusk
            t @ DAY_TIME..DUSK_END => Some(max_radius * (DUSK_END - t) as i32 / DUSK_TIME as i32),
            _ => Some(0),
        };

        self.state.time += 1;
    }
}
