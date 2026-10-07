//! What an episode reports: deaths, bites, and per-life achievements.

/// Declares the achievements with their metric names side by side, so the two
/// can't fall out of step.
macro_rules! achievements {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        /// Craftax-style milestones: each counts at most once per life, so the
        /// metric reads as how far agents get rather than how often they
        /// repeat a step.
        #[derive(Debug, Clone, Copy)]
        pub(super) enum Achievement {
            $($variant),+
        }

        impl Achievement {
            /// Every achievement, each at its `as usize` index.
            pub(super) const ALL: &[Achievement] = &[$(Achievement::$variant),+];

            /// The name its metric is reported under.
            pub(super) fn name(self) -> &'static str {
                match self {
                    $(Achievement::$variant => $name),+
                }
            }
        }
    };
}

achievements! {
    CollectStick => "collect_stick",
    CollectStone => "collect_stone",
    CollectWood => "collect_wood",
    CollectBerry => "collect_berry",
    CollectCarrot => "collect_carrot",
    CollectGrass => "collect_grass",
    ChopTree => "chop_tree",
    ClearBush => "clear_bush",
    HarvestGrass => "harvest_grass",
    MakeAxe => "make_axe",
    MakeCampfire => "make_campfire",
    MakeTorch => "make_torch",
    PlaceFire => "place_fire",
    RefuelFire => "refuel_fire",
    CookBerry => "cook_berry",
    CookCarrot => "cook_carrot",
    EatBerry => "eat_berry",
    EatCookedBerry => "eat_cooked_berry",
    EatCarrot => "eat_carrot",
    EatCookedCarrot => "eat_cooked_carrot",
}

#[derive(Debug, Default, Clone)]
pub(super) struct SurvivalMetrics {
    pub(super) deaths: f64,
    pub(super) spider_bites: f64,
    /// Unlocks of each achievement, in `Achievement` order.
    pub(super) achievements: [f64; Achievement::ALL.len()],
}

impl SurvivalMetrics {
    /// The counts as averages over `agents`.
    pub(super) fn per_agent(&self, agents: usize) -> serde_json::Value {
        let agents = agents.max(1) as f64;
        let achievements: serde_json::Map<_, _> = Achievement::ALL
            .iter()
            .zip(self.achievements)
            .map(|(a, count)| (a.name().to_owned(), serde_json::Value::from(count / agents)))
            .collect();
        serde_json::json!({
            "deaths": self.deaths / agents,
            "spider_bites": self.spider_bites / agents,
            "achievements": achievements,
        })
    }
}
