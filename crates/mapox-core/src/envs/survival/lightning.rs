use crate::envs::{
    common::{
        Position,
        fov::{self, ViewTile},
    },
    survival::{Survival, fire::Active},
};

pub(super) struct LightEmitter {
    pub radius: i32,
}

impl Survival {
    pub(super) fn prepare_lights(&mut self) {
        let (world, lighting, render_map) = (
            &mut self.state.world,
            &mut self.state.lighting,
            &self.state.render_map,
        );
        let is_dusk = self.state.dusk.is_some();
        lighting.fill(!is_dusk);

        if is_dusk {
            for (&position, light, active) in
                world.query_mut::<(&Position, &LightEmitter, Option<&Active>)>()
            {
                if !Active::is_on(active) {
                    continue;
                }
                let in_reach = |cell: Position| fov::within(cell - position, light.radius);
                fov::shadowcast(
                    position,
                    lighting.dim(),
                    |cell| in_reach(cell) && !render_map[cell.idx()].opaque(),
                    |cell| {
                        if in_reach(cell) {
                            lighting[cell.idx()] = true;
                        }
                    },
                );
            }
        }
    }
}
