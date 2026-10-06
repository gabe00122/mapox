//! Day and night: the cycle, and how far it lets agents see.

use crate::envs::common::Position;

use super::{Survival, tiles::within};

impl Survival {
    fn cycle_length(&self) -> usize {
        (self.config.day_length + self.config.night_length) as usize
    }

    pub(super) fn is_night(&self, time: usize) -> bool {
        time % self.cycle_length() >= self.config.day_length as usize
    }

    /// Whether `time` is the first step of a night.
    pub(super) fn is_nightfall(&self, time: usize) -> bool {
        time % self.cycle_length() == self.config.day_length as usize
    }

    /// How far agents see around themselves at `time`, firelight aside: no
    /// limit by day, `night_vision_radius` by night, and in between, through
    /// dusk, a radius shrinking a little every step from one that takes in the
    /// whole view.
    pub(super) fn vision_radius(&self, time: usize) -> Option<i32> {
        let night = self.config.night_vision_radius;
        let phase = time % self.cycle_length();
        let day = self.config.day_length as usize;
        let dusk = self.config.dusk_length as usize;
        if phase >= day {
            return Some(night);
        }
        // steps of daylight left, this one included
        let left = day - phase;
        if left > dusk {
            return None;
        }

        // the smallest disc around the agent that covers its whole view
        let corner = Position::new(self.config.view_width / 2, self.config.view_height / 2);
        let full = (0..)
            .find(|&r| within(corner, r))
            .expect("some disc covers the view");
        Some(night + (full - night) * left as i32 / (dusk as i32 + 1))
    }
}
