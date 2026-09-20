// Atlas bytes have egui's premultiplied gamma-space representation.
@group(0) @binding(0) var atlas: texture_2d<f32>;
// One RGBA8Uint texel per environment cell: atlas col, row, unseen, unused.
@group(0) @binding(1) var tile_indices: texture_2d<u32>;

struct Geometry {
    // Unclipped, unrounded callback rectangle in physical pixels.
    rect: vec4<f32>,
    // Source origin x/y, tile stride, tile size, from Tileset::source.
    atlas_layout: vec4<u32>,
};
@group(0) @binding(2) var<uniform> geometry: Geometry;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex], 0.0, 1.0);
}

struct Tile {
    color: vec4<f32>,
    unseen: bool,
};

fn sample_tile(position: vec2<f32>) -> Tile {
    let dimensions = textureDimensions(tile_indices);
    let grid_position = (position - geometry.rect.xy) / geometry.rect.zw
        * vec2<f32>(dimensions);
    if any(grid_position < vec2<f32>(0.0))
        || any(grid_position >= vec2<f32>(dimensions)) {
        discard;
    }
    let screen_cell = vec2<u32>(grid_position);
    // Flip only the environment grid. Each sprite retains its top-down rows.
    let cell = vec2<i32>(i32(screen_cell.x), i32(dimensions.y - 1u - screen_cell.y));
    let tile = textureLoad(tile_indices, cell, 0);
    let tile_size = geometry.atlas_layout.w;
    let texel = min(
        vec2<u32>(fract(grid_position) * f32(tile_size)),
        vec2<u32>(tile_size - 1u),
    );
    let source = geometry.atlas_layout.xy + tile.xy * geometry.atlas_layout.z;
    return Tile(textureLoad(atlas, vec2<i32>(source + texel), 0), tile.z != 0u);
}

// Same transfer function and framebuffer distinction as egui-wgpu's egui.wgsl.
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// Color32::from_rgba_premultiplied(41, 41, 41, 110), not straight-alpha gray.
const FOG: vec4<f32> = vec4<f32>(41.0, 41.0, 41.0, 110.0) / 255.0;

@fragment
fn fs_main_gamma_framebuffer(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = sample_tile(position.xy);
    if tile.unseen {
        return FOG + tile.color * (1.0 - FOG.a);
    }
    return tile.color;
}

@fragment
fn fs_main_linear_framebuffer(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = sample_tile(position.xy);
    let color = vec4<f32>(linear_from_gamma_rgb(tile.color.rgb), tile.color.a);
    if tile.unseen {
        // An sRGB attachment blends in linear space. Convert each egui draw's
        // premultiplied color before combining, just as two separate draws do.
        let fog = vec4<f32>(linear_from_gamma_rgb(FOG.rgb), FOG.a);
        return fog + color * (1.0 - FOG.a);
    }
    return color;
}
