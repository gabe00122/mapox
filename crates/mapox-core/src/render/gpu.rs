//! One draw for the tile grid and its fog, with dirty-only tile uploads.

use std::{num::NonZeroU64, sync::Arc};

use eframe::egui_wgpu::{
    Callback, CallbackResources, CallbackTrait, RenderState, ScreenDescriptor,
};
use egui::{PaintCallbackInfo, Painter, Rect};

use super::tileset::{TILESET_COLS, TILESET_ROWS, Tileset};

/// GPU resources belong to this renderer and its outstanding paint callbacks,
/// not to a process-global or egui-global singleton.
pub(crate) struct TilemapRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: Arc<TilemapPipeline>,
    grid: Option<Arc<TilemapGrid>>,
    staging: Vec<u8>,
}

struct TilemapPipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    atlas: wgpu::TextureView,
    // Source origin x/y, tile stride, tile size; shared with the WGSL uniform.
    atlas_layout: [u32; 4],
}

struct TilemapGrid {
    pipeline: Arc<TilemapPipeline>,
    indices: wgpu::Texture,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct TilemapCallback {
    grid: Arc<TilemapGrid>,
    rect: Rect,
}

impl TilemapRenderer {
    pub(crate) fn new(state: &RenderState) -> Self {
        let device = &state.device;
        let image = Tileset::decode();
        let (width, height) = image.dimensions();
        let limit = device.limits().max_texture_dimension_2d;
        assert!(
            width <= limit && height <= limit,
            "tilemap atlas {width}x{height} exceeds wgpu max_texture_dimension_2d ({limit})"
        );
        let mut pixels = image.into_raw();
        // Match ColorImage::from_rgba_unmultiplied, used by Tileset::embedded.
        // Rgba8Unorm deliberately keeps the atlas in egui's gamma space.
        for pixel in pixels.as_chunks_mut::<4>().0 {
            if pixel[3] != 255 {
                let color =
                    egui::Color32::from_rgba_unmultiplied(pixel[0], pixel[1], pixel[2], pixel[3]);
                pixel.copy_from_slice(&color.to_array());
            }
        }
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mapox tile atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        state.queue.write_texture(
            atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            atlas.size(),
        );
        let atlas = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mapox tilemap bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(32),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mapox tilemap pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mapox tilemap shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tilemap.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mapox tilemap pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if state.target_format.is_srgb() {
                    "fs_main_linear_framebuffer"
                } else {
                    "fs_main_gamma_framebuffer"
                }),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: state.target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let first_tile = Tileset::source(0, 0);
        let tile_stride = Tileset::source(1, 0).min.x - first_tile.min.x;
        Self {
            device: device.clone(),
            queue: state.queue.clone(),
            pipeline: Arc::new(TilemapPipeline {
                pipeline,
                layout,
                atlas,
                atlas_layout: [
                    first_tile.min.x as u32,
                    first_tile.min.y as u32,
                    tile_stride as u32,
                    first_tile.width() as u32,
                ],
            }),
            grid: None,
            staging: Vec::new(),
        }
    }

    /// Cells use environment coordinates (y up); atlas rows remain top-down.
    /// Call only after a map, visibility or view change.
    pub(crate) fn update(
        &mut self,
        cols: usize,
        rows: usize,
        mut cell: impl FnMut(usize, usize) -> ((u32, u32), bool),
    ) {
        if cols == 0 || rows == 0 {
            self.grid = None;
            self.staging.clear();
            return;
        }
        let limits = self.device.limits();
        let limit = limits.max_texture_dimension_2d as usize;
        assert!(
            cols <= limit && rows <= limit,
            "tilemap grid {cols}x{rows} exceeds wgpu max_texture_dimension_2d ({limit})"
        );
        let byte_len = cols
            .checked_mul(rows)
            .and_then(|cells| cells.checked_mul(4))
            .expect("tilemap index upload exceeds addressable memory");
        assert!(
            byte_len as u64 <= limits.max_buffer_size,
            "tilemap index upload ({byte_len} bytes) exceeds wgpu max_buffer_size ({})",
            limits.max_buffer_size
        );
        self.staging.resize(byte_len, 0);
        for y in 0..rows {
            for x in 0..cols {
                let ((atlas_col, atlas_row), visible) = cell(x, y);
                assert!(
                    atlas_col < TILESET_COLS && atlas_row < TILESET_ROWS,
                    "tilemap cell ({x}, {y}) selects atlas tile ({atlas_col}, {atlas_row}) outside {TILESET_COLS}x{TILESET_ROWS}"
                );
                let offset = (y * cols + x) * 4;
                self.staging[offset..offset + 4].copy_from_slice(&[
                    atlas_col as u8,
                    atlas_row as u8,
                    u8::from(!visible),
                    0,
                ]);
            }
        }
        if self.grid.as_ref().is_none_or(|grid| {
            grid.indices.width() != cols as u32 || grid.indices.height() != rows as u32
        }) {
            self.grid = Some(Arc::new(TilemapGrid::new(
                &self.device,
                &self.queue,
                Arc::clone(&self.pipeline),
                cols as u32,
                rows as u32,
            )));
        }
        let grid = self.grid.as_ref().expect("nonempty tilemap has a texture");
        self.queue.write_texture(
            grid.indices.as_image_copy(),
            &self.staging,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cols as u32 * 4),
                rows_per_image: Some(rows as u32),
            },
            grid.indices.size(),
        );
    }

    /// At most once per frame: every callback on a grid shares one uniform
    /// buffer, so a second rect would overwrite the first before either draws.
    pub(crate) fn paint(&self, painter: &Painter, rect: Rect) {
        if let Some(grid) = &self.grid
            && rect.is_positive()
            && rect.is_finite()
        {
            painter.add(Callback::new_paint_callback(
                rect,
                TilemapCallback {
                    grid: Arc::clone(grid),
                    rect,
                },
            ));
        }
    }
}

impl TilemapGrid {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: Arc<TilemapPipeline>,
        cols: u32,
        rows: u32,
    ) -> Self {
        let indices = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mapox tile indices and fog"),
            size: wgpu::Extent3d {
                width: cols,
                height: rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let indices_view = indices.create_view(&wgpu::TextureViewDescriptor::default());
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mapox tilemap geometry"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 16, bytemuck::bytes_of(&pipeline.atlas_layout));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mapox tilemap bind group"),
            layout: &pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pipeline.atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&indices_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            indices,
            uniforms,
            bind_group,
        }
    }
}

impl CallbackTrait for TilemapCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Egui rounds/clamps the viewport, but the original fractional map
        // geometry must survive both clipping and sub-texel window sizes.
        let scale = screen.pixels_per_point;
        let rect = [
            self.rect.min.x * scale,
            self.rect.min.y * scale,
            self.rect.width() * scale,
            self.rect.height() * scale,
        ];
        queue.write_buffer(&self.grid.uniforms, 0, bytemuck::bytes_of(&rect));
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &CallbackResources,
    ) {
        // Egui already supplies the callback viewport and the painter's
        // scissor rectangle. Leave both intact, especially under clipping.
        render_pass.set_pipeline(&self.grid.pipeline.pipeline);
        render_pass.set_bind_group(0, &self.grid.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
