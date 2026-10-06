//! Reads the safetensors bundle written by jaxrl's
//! `scripts/export_burn_policy.py`: raw flax-layout tensors under dotted nnx
//! state paths, plus JSON configs in the metadata. Tensors may be stored as
//! f32, f16 or bf16 (half precision halves the download for the web build);
//! all are widened to f32 here, so inference always computes in f32. All
//! layout conversion to burn conventions happens here too:
//!
//! - `nnx.Linear` kernels are `[in, out]`, which is also burn's orientation.
//! - the fused `nnx.LinearGeneral` qkv kernel is `[d_model, heads + 2 *
//!   kv_heads, head_dim]`, query heads then key heads then value heads; it
//!   flattens contiguously to `[d_model, (heads + 2 * kv_heads) * head_dim]`
//!   and is split by column into the three projections. The out kernel
//!   `[heads, head_dim, d_model]` flattens straight to the 2d matmul shape.
//! - `nnx.Conv` kernels are `[kh, kw, in, out]` (NHWC), permuted to burn's
//!   `[out, in, kh, kw]` (NCHW).

use std::collections::HashMap;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

use crate::config::PolicyMetadata;
use crate::model::{
    Attention, ConvLayer, Embedder, FeedForward, GridCnnEncoder, Linear, RmsNorm, TransformerActor,
    TransformerLayer,
};

pub const FORMAT: &str = "mapox-burn-v1";

pub type LoadError = Box<dyn std::error::Error + Send + Sync>;

pub struct LoadedPolicy<B: Backend> {
    pub model: TransformerActor<B>,
    pub meta: PolicyMetadata,
}

pub fn load_policy<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedPolicy<B>, LoadError> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("reading {}: {e}", path.as_ref().display()))?;
    load_policy_bytes(&bytes, device)
}

/// The bundle already in memory, for callers without a filesystem: the wasm
/// build fetches it over http and hands the bytes straight over.
pub fn load_policy_bytes<B: Backend>(
    bytes: &[u8],
    device: &B::Device,
) -> Result<LoadedPolicy<B>, LoadError> {
    let meta = parse_metadata(bytes)?;
    let tensors = SafeTensors::deserialize(bytes)?;
    let model = build_model(&meta, &Store { tensors, device })?;
    Ok(LoadedPolicy { model, meta })
}

pub fn parse_metadata(bytes: &[u8]) -> Result<PolicyMetadata, LoadError> {
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let meta: &HashMap<String, String> = header
        .metadata()
        .as_ref()
        .ok_or("safetensors file has no metadata; was it written by export_burn_policy.py?")?;

    let get = |key: &str| -> Result<&String, LoadError> {
        meta.get(key)
            .ok_or_else(|| format!("metadata missing {key:?}").into())
    };

    let format = get("format")?;
    if format != FORMAT {
        return Err(format!("unsupported format {format:?}, expected {FORMAT:?}").into());
    }

    let obs_shape: Vec<usize> = serde_json::from_str(get("obs_shape")?)?;
    let obs_shape: [usize; 3] = obs_shape
        .try_into()
        .map_err(|s: Vec<usize>| format!("obs_shape has {} dims, expected 3", s.len()))?;

    Ok(PolicyMetadata {
        model: serde_json::from_str(get("model_config")?)
            .map_err(|e| format!("parsing model_config: {e}"))?,
        obs_shape,
        obs_max_value: serde_json::from_str(get("obs_max_value")?)?,
        action_dim: get("action_dim")?.parse()?,
        num_tasks: get("num_tasks")?.parse()?,
        max_seq_length: get("max_seq_length")?.parse()?,
        env_config: meta.get("env_config").cloned(),
        source: get("source")?.clone(),
    })
}

/// Tensor lookup with dtype/shape checking and flax→burn layout conversion.
struct Store<'a, B: Backend> {
    tensors: SafeTensors<'a>,
    device: &'a B::Device,
}

impl<'a, B: Backend> Store<'a, B> {
    fn raw(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), LoadError> {
        let view = self
            .tensors
            .tensor(name)
            .map_err(|_| format!("checkpoint is missing tensor {name:?}"))?;
        let bytes = view.data();
        let data = match view.dtype() {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32())
                .collect(),
            dtype => return Err(format!("{name}: expected f32, f16 or bf16, got {dtype:?}").into()),
        };
        Ok((data, view.shape().to_vec()))
    }

    /// Load with the stored shape reinterpreted as `shape` (contiguous
    /// reshape; element counts must match).
    fn reshaped<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
    ) -> Result<Tensor<B, D>, LoadError> {
        let (data, stored) = self.raw(name)?;
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "{name}: stored shape {stored:?} has {} elements, expected {expected} for {shape:?}",
                data.len()
            )
            .into());
        }
        Ok(Tensor::from_data(TensorData::new(data, shape), self.device))
    }

    /// Column blocks of a `[rows, sum(widths)]` matrix as separate
    /// contiguous `[rows, width]` tensors, split on the host so the matmuls
    /// never see a strided view.
    fn split_columns(
        &self,
        name: &str,
        rows: usize,
        widths: &[usize],
    ) -> Result<Vec<Tensor<B, 2>>, LoadError> {
        let (data, stored) = self.raw(name)?;
        let cols: usize = widths.iter().sum();
        if data.len() != rows * cols {
            return Err(format!(
                "{name}: stored shape {stored:?} has {} elements, expected {rows}x{cols}",
                data.len()
            )
            .into());
        }
        let mut start = 0;
        Ok(widths
            .iter()
            .map(|&width| {
                let block: Vec<f32> = data
                    .chunks_exact(cols)
                    .flat_map(|row| &row[start..start + width])
                    .copied()
                    .collect();
                start += width;
                Tensor::from_data(TensorData::new(block, [rows, width]), self.device)
            })
            .collect())
    }

    fn exact<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
    ) -> Result<Tensor<B, D>, LoadError> {
        let (_, stored) = self.raw(name)?;
        if stored != shape {
            return Err(format!("{name}: stored shape {stored:?}, expected {shape:?}").into());
        }
        self.reshaped(name, shape)
    }

    /// A `use_bias=True` `nnx.Linear`. Bias-free kernels load via [`Self::kernel`].
    fn linear(
        &self,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Linear<B>, LoadError> {
        Ok(Linear {
            weight: self.kernel(name, in_features, out_features)?,
            bias: self.exact(&format!("{name}.bias"), [out_features])?,
        })
    }

    fn kernel(
        &self,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Tensor<B, 2>, LoadError> {
        self.exact(&format!("{name}.kernel"), [in_features, out_features])
    }

    fn norm(&self, name: &str, features: usize) -> Result<RmsNorm<B>, LoadError> {
        Ok(RmsNorm {
            scale: self.exact(&format!("{name}.scale"), [features])?,
        })
    }
}

fn build_model<B: Backend>(
    meta: &PolicyMetadata,
    store: &Store<'_, B>,
) -> Result<TransformerActor<B>, LoadError> {
    let config = &meta.model;
    let hidden = config.hidden_features;

    // like the jax encoder, which embeds channel 0 only
    if meta.obs_shape[2] != 1 {
        return Err(format!(
            "the obs encoder embeds a single channel, obs has {}",
            meta.obs_shape[2]
        )
        .into());
    }

    let cnn = &config.obs_encoder;
    if cnn.kernels.len() != cnn.strides.len() || cnn.kernels.len() != cnn.channels.len() + 1 {
        return Err(format!(
            "cnn config mismatch: {} kernels, {} strides, {} channels (+ output)",
            cnn.kernels.len(),
            cnn.strides.len(),
            cnn.channels.len()
        )
        .into());
    }
    let num_classes: usize = meta.obs_max_value.iter().sum();
    let embedding = store.exact("obs_encoder.embedding", [num_classes, cnn.embedding_dim])?;
    let mut conv_layers = Vec::new();
    let mut in_channels = cnn.embedding_dim;
    let out_channels = cnn.channels.iter().copied().chain([hidden]);
    for (i, ((kernel, stride), out)) in cnn
        .kernels
        .iter()
        .zip(&cnn.strides)
        .zip(out_channels)
        .enumerate()
    {
        let name = format!("obs_encoder.layers.{i}");
        let weight = store
            .exact(
                &format!("{name}.kernel"),
                [kernel[0], kernel[1], in_channels, out],
            )?
            .permute([3, 2, 0, 1]);
        conv_layers.push(ConvLayer {
            weight,
            bias: store.exact(&format!("{name}.bias"), [out])?,
            stride: *stride,
        });
        in_channels = out;
    }
    let obs_encoder = GridCnnEncoder {
        embedding,
        layers: conv_layers,
    };

    let mut layers = Vec::new();
    for i in 0..config.num_layers {
        let name = format!("layers.{i}");

        let history = match &config.layer.history {
            None => None,
            Some(attn) => {
                let (heads, kv_heads, head_dim) =
                    (attn.num_heads, attn.num_kv_heads, attn.head_dim);
                if heads % kv_heads != 0 {
                    return Err(
                        format!("num_heads {heads} not divisible by kv heads {kv_heads}").into(),
                    );
                }
                let (rope_sin, rope_cos) = Attention::build_rope_tables(
                    meta.max_seq_length,
                    head_dim,
                    attn.rope_max_wavelength,
                    store.device,
                );
                let [query_proj, key_proj, value_proj]: [Tensor<B, 2>; 3] = store
                    .split_columns(
                        &format!("{name}.history.qkv_proj.kernel"),
                        hidden,
                        &[heads * head_dim, kv_heads * head_dim, kv_heads * head_dim],
                    )?
                    .try_into()
                    .expect("three column blocks");
                let attention = Attention {
                    query_proj,
                    key_proj,
                    value_proj,
                    out_proj: store.reshaped(
                        &format!("{name}.history.out.kernel"),
                        [heads * head_dim, hidden],
                    )?,
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                    max_seq_length: meta.max_seq_length,
                    rope_sin,
                    rope_cos,
                    causal_mask: Attention::build_causal_mask(meta.max_seq_length, store.device),
                };
                Some((
                    store.norm(&format!("{name}.history_norm"), hidden)?,
                    attention,
                ))
            }
        };

        let ff_size = config.layer.feed_forward.size;
        layers.push(TransformerLayer {
            history,
            ffn_norm: store.norm(&format!("{name}.ffn_norm"), hidden)?,
            ffn: FeedForward {
                up_proj: store.kernel(&format!("{name}.ffn.up_proj"), hidden, ff_size)?,
                up_gate: store.kernel(&format!("{name}.ffn.up_gate"), hidden, ff_size)?,
                down_proj: store.kernel(&format!("{name}.ffn.down_proj"), ff_size, hidden)?,
            },
        });
    }

    // `value_mlp` and `value_head.*` are in the bundle and deliberately left
    // there: the critic is a training artifact and acting never reads it.
    Ok(TransformerActor {
        obs_encoder,
        reward_encoder: store.linear("reward_encoder", 1, hidden)?,
        action_embedder: Embedder {
            table: store.exact("action_embedder.embedding_table", [meta.action_dim, hidden])?,
        },
        // jax only builds it for multitask runs
        task_embedder: if meta.num_tasks > 1 {
            Some(Embedder {
                table: store.exact("task_embedder.embedding_table", [meta.num_tasks, hidden])?,
            })
        } else {
            None
        },
        layers,
        output_norm: store.norm("output_norm", hidden)?,
        max_seq_length: meta.max_seq_length,
        action_dim: meta.action_dim,
        device: store.device.clone(),
    })
}
