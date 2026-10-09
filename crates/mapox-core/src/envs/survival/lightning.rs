use crate::envs::{
    common::{
        Position,
        fov::{self, ViewTile},
    },
    survival::Survival,
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
        lighting.fill(false);

        for (&position, light) in world.query_mut::<(&Position, &LightEmitter)>() {
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
