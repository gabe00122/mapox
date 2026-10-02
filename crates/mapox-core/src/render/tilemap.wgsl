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

// Set at pipeline creation: an sRGB attachment takes linear colors.
override LINEAR_FRAMEBUFFER: bool;

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

// One art pixel of the whole map, addressed top-down in sprite texels and
// clamped to the map, so the filter's border neighbours repeat the edge.
fn load_texel(texel: vec2<i32>) -> Tile {
    let tile_size = vec2<i32>(i32(geometry.atlas_layout.w));
    let dimensions = vec2<i32>(textureDimensions(tile_indices));
    let clamped = clamp(texel, vec2<i32>(0), dimensions * tile_size - 1);
    let screen_cell = clamped / tile_size;
    // Flip only the environment grid. Each sprite retains its top-down rows.
    let cell = vec2<i32>(screen_cell.x, dimensions.y - 1 - screen_cell.y);
    let tile = textureLoad(tile_indices, cell, 0);
    let source = vec2<i32>(geometry.atlas_layout.xy + tile.xy * geometry.atlas_layout.z);
    return Tile(textureLoad(atlas, source + clamped % tile_size, 0), tile.z != 0u);
}

// The four art pixels around a fragment and how much of the fragment the
// lower-right ones cover. Away from a texel edge the weights saturate and
// this is nearest sampling; across one it is the pixel's exact box coverage,
// so a fractional scale draws every texel at the same apparent width.
struct Footprint {
    texel: vec2<i32>,
    weight: vec2<f32>,
};

fn footprint(position: vec2<f32>) -> Footprint {
    let dimensions = vec2<f32>(textureDimensions(tile_indices));
    let grid_position = (position - geometry.rect.xy) / geometry.rect.zw * dimensions;
    if any(grid_position < vec2<f32>(0.0)) || any(grid_position >= dimensions) {
        discard;
    }
    let tile_size = vec2<f32>(f32(geometry.atlas_layout.w));
    // Shift by half a texel so `base` is the nearest texel centre up-left.
    let art = grid_position * tile_size - 0.5;
    let art_per_pixel = tile_size * dimensions / geometry.rect.zw;
    let base = floor(art);
    let weight = clamp((art - base - 0.5) / art_per_pixel + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    return Footprint(vec2<i32>(base), weight);
}

// Same transfer function and framebuffer distinction as egui-wgpu's egui.wgsl.
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// Unseen cells keep their art but lose most colour and brightness.
const UNSEEN_DESATURATION: f32 = 0.7;
const UNSEEN_BRIGHTNESS: f32 = 0.55;

fn shade(texel: vec2<i32>) -> vec4<f32> {
    let tile = load_texel(texel);
    var color = tile.color;
    if tile.unseen {
        // Done in gamma space so the look is the same on either framebuffer.
        // Scaling premultiplied rgb without alpha keeps it premultiplied.
        let grey = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let muted = mix(color.rgb, vec3<f32>(grey), UNSEEN_DESATURATION);
        color = vec4<f32>(muted * UNSEEN_BRIGHTNESS, color.a);
    }
    if LINEAR_FRAMEBUFFER {
        // An sRGB attachment takes linear colors, as in egui-wgpu.
        color = vec4<f32>(linear_from_gamma_rgb(color.rgb), color.a);
    }
    return color;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let f = footprint(position.xy);
    // Shaded colors are premultiplied, so a plain lerp mixes them correctly.
    let top = mix(shade(f.texel), shade(f.texel + vec2<i32>(1, 0)), f.weight.x);
    let bottom = mix(shade(f.texel + vec2<i32>(0, 1)), shade(f.texel + vec2<i32>(1, 1)), f.weight.x);
    return mix(top, bottom, f.weight.y);
}
