use crate::envs::{
    common::{
        Position,
        fov::{self, ViewTile},
    },
    survival::Survival,
};

impl Survival {
    pub(super) fn prepare_lights(&mut self) {
        let state = &mut self.state;

        state.lighting.fill(false);
        let (lighting, render_map) = (&mut state.lighting, &state.render_map);
        let light = Position::new(10, 10);
        // past the light's reach a cell stays dark and blocks, and so is all
        // behind it
        let in_reach = |cell: Position| fov::within(cell - light, 5);
        fov::shadowcast(
            light,
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
