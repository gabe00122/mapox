//! The actor half of the jaxrl `TransformerActorCritic` rebuilt on burn
//! tensors, decode-only: one env step per call with a kv-cache carry, the same
//! path `model.sample_actions` takes in the trainer.
//!
//! Everything computes in f32. The jax model usually computes in bfloat16
//! with f32 params; f32 here is strictly more precise, so outputs match the
//! f32-referenced export within float error and the bf16 demo approximately.
//!
//! Parity notes, mirrored from the jax modules:
//! - `gelu` is the tanh approximation (`jax.nn.gelu(approximate=True)`),
//!   which burn calls `gelu_approximate`; used between cnn layers and in feed
//!   forward.
//! - flax `RMSNorm` defaults to `epsilon = 1e-6`.
//! - the embedder scales lookups by `sqrt(hidden)` and its table doubles as
//!   the action head (`decode` is a matmul with the transpose).
//! - rope is the split-half form (not interleaved), positions are the raw
//!   step count, and the kv cache holds one slot per step up to
//!   `max_seq_length`, attending to the `time + 1` written so far. the trainer
//!   gets that from `is_causal=True` over a whole sequence; decoding one step
//!   at a time leaves a single query row, where the same `tril` reduces to a
//!   mask over the slots not written yet.
//! - grouped-query attention groups *consecutive* query heads per kv head,
//!   matching `jax.nn.dot_product_attention`'s `(B, T, K, G, H)` reshape.

use burn::tensor::activation::{gelu_approximate, log_softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use ndarray::{ArrayView2, ArrayView4};

/// flax `RMSNorm`'s default epsilon.
const NORM_EPS: f64 = 1e-6;

/// `max_wavelength ** (2i / head_dim)`, the rope frequency denominator.
fn rope_timescale(max_wavelength: f64, i: usize, head_dim: usize) -> f64 {
    max_wavelength.powf(2.0 * i as f64 / head_dim as f64)
}

/// `nnx.Linear` with `use_bias`: weight is `[in, out]` exactly as flax stores
/// the kernel. The bias-free projections are plain matmuls, so they hold their
/// kernels directly rather than going through here.
pub struct Linear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> Linear<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.clone()) + self.bias.clone().unsqueeze::<2>()
    }
}

/// flax `RMSNorm`.
pub struct RmsNorm<B: Backend> {
    pub scale: Tensor<B, 1>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean_sq = x.clone().powf_scalar(2.0).mean_dim(1);
        x * (mean_sq + NORM_EPS).sqrt().recip() * self.scale.clone().unsqueeze::<2>()
    }
}

/// `GLUBlock`: `down(gelu(up(x)) * up_gate(x))`, all three kernels `[in, out]`
/// and bias-free.
pub struct FeedForward<B: Backend> {
    pub up_proj: Tensor<B, 2>,
    pub up_gate: Tensor<B, 2>,
    pub down_proj: Tensor<B, 2>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = gelu_approximate(x.clone().matmul(self.up_proj.clone()))
            * x.matmul(self.up_gate.clone());
        hidden.matmul(self.down_proj.clone())
    }
}

/// `Embedder`: lookup scaled by `sqrt(features)`; `decode` shares the table.
pub struct Embedder<B: Backend> {
    pub table: Tensor<B, 2>, // [vocab, features]
}

impl<B: Backend> Embedder<B> {
    pub fn encode(&self, ids: &[u16], device: &B::Device) -> Tensor<B, 2> {
        let [vocab, features] = self.table.dims();
        let indices: Vec<i32> = ids
            .iter()
            .map(|&id| {
                assert!((id as usize) < vocab, "embedding id {id} out of range");
                id as i32
            })
            .collect();
        let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [ids.len()]), device);
        self.table.clone().select(0, indices) * (features as f64).sqrt()
    }

    pub fn decode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.table.clone().transpose())
    }
}

/// One `nnx.Conv` with VALID padding, weight already in `[out, in, kh, kw]`.
pub struct ConvLayer<B: Backend> {
    pub weight: Tensor<B, 4>,
    pub bias: Tensor<B, 1>,
    pub stride: [usize; 2],
}

/// `GridCnnObsEncoder`: one-hot concat over channels, then strided convs with
/// gelu between (not after) them, flattened to the hidden size.
pub struct GridCnnEncoder<B: Backend> {
    pub layers: Vec<ConvLayer<B>>,
    pub one_hot_sizes: Vec<usize>,
}

impl<B: Backend> GridCnnEncoder<B> {
    /// `obs` is `(agents, view_w, view_h, channels)` of vocab ids.
    pub fn forward(&self, obs: ArrayView4<'_, u16>, device: &B::Device) -> Tensor<B, 2> {
        let (agents, width, height, channels) = obs.dim();
        assert_eq!(channels, self.one_hot_sizes.len(), "obs channel mismatch");

        let mut offsets = Vec::with_capacity(channels);
        let mut total = 0usize;
        for &size in &self.one_hot_sizes {
            offsets.push(total);
            total += size;
        }

        // one-hot straight into NCHW so no permute is needed; jax builds NHWC
        // (w, h, classes) and convolves channels-last, same math
        let mut buffer = vec![0f32; agents * total * width * height];
        for agent in 0..agents {
            for x in 0..width {
                for y in 0..height {
                    for channel in 0..channels {
                        let id = obs[[agent, x, y, channel]] as usize;
                        assert!(
                            id < self.one_hot_sizes[channel],
                            "obs id {id} exceeds channel {channel} vocab"
                        );
                        let class = offsets[channel] + id;
                        buffer[((agent * total + class) * width + x) * height + y] = 1.0;
                    }
                }
            }
        }

        let mut t = Tensor::<B, 4>::from_data(
            TensorData::new(buffer, [agents, total, width, height]),
            device,
        );
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            t = burn::tensor::module::conv2d(
                t,
                layer.weight.clone(),
                Some(layer.bias.clone()),
                burn::tensor::ops::ConvOptions::new(layer.stride, [0, 0], [1, 1], 1),
            );
            if i < last {
                t = gelu_approximate(t);
            }
        }

        // jax flattens (w, h, c); permute back from NCHW so the order matches
        // (a no-op for the usual 1x1 final spatial size)
        let [batch, c, w, h] = t.dims();
        t.permute([0, 2, 3, 1]).reshape([batch, w * h * c])
    }
}

/// The attention kv cache: one slot per step, held in the layout `attention`
/// wants so a step never permutes it (see `forward`). Sized for a whole
/// `max_seq_length` episode, which is as far as a carry ever runs.
pub struct KvCache<B: Backend> {
    pub key: Tensor<B, 4>, // [batch, kv_heads, max_seq, head_dim]
    pub value: Tensor<B, 4>,
}

pub struct Attention<B: Backend> {
    pub query_proj: Tensor<B, 2>, // [d_model, heads * head_dim]
    pub key_proj: Tensor<B, 2>,   // [d_model, kv_heads * head_dim]
    pub value_proj: Tensor<B, 2>,
    pub out_proj: Tensor<B, 2>, // [heads * head_dim, d_model]
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_length: usize,
    /// `[max_seq_length, head_dim / 2]` rope tables, resident on the device
    /// so a step slices a row instead of uploading freshly built ones.
    pub rope_sin: Tensor<B, 2>,
    pub rope_cos: Tensor<B, 2>,
    /// `[max_seq_length, max_seq_length]`, row `time` true wherever the cache
    /// slot is still unwritten. The single-query-row specialization of the
    /// `tril` mask `jax.nn.dot_product_attention(is_causal=True)` builds in
    /// the trainer, kept as data so the kernel sees one fixed shape.
    pub causal_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> Attention<B> {
    /// Split-half rope sin/cos for positions `0..max_seq_length`.
    pub fn build_rope_tables(
        max_seq_length: usize,
        head_dim: usize,
        max_wavelength: f64,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let half = head_dim / 2;
        let (sin, cos): (Vec<f32>, Vec<f32>) = (0..max_seq_length)
            .flat_map(|position| (0..half).map(move |i| (position, i)))
            .map(|(position, i)| {
                let angle = position as f64 / rope_timescale(max_wavelength, i, head_dim);
                (angle.sin() as f32, angle.cos() as f32)
            })
            .unzip();
        (
            Tensor::from_data(TensorData::new(sin, [max_seq_length, half]), device),
            Tensor::from_data(TensorData::new(cos, [max_seq_length, half]), device),
        )
    }
    /// Row `time` masks every cache slot not written yet, so attending over
    /// the whole buffer matches attending over the written prefix.
    pub fn build_causal_mask(max_seq_length: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
        let unwritten: Vec<bool> = (0..max_seq_length)
            .flat_map(|time| (0..max_seq_length).map(move |slot| slot > time))
            .collect();
        Tensor::from_data(
            TensorData::new(unwritten, [max_seq_length, max_seq_length]),
            device,
        )
    }

    pub fn init_cache(&self, batch: usize, device: &B::Device) -> KvCache<B> {
        let shape = [batch, self.num_kv_heads, self.max_seq_length, self.head_dim];
        KvCache {
            key: Tensor::zeros(shape, device),
            value: Tensor::zeros(shape, device),
        }
    }

    /// Split-half rope at one position, sliced from the resident tables.
    /// `time` is always inside them: [`BurnPolicy`](crate::BurnPolicy) cycles
    /// the carry at `max_seq_length` at the latest.
    fn rope(&self, x: Tensor<B, 3>, time: usize) -> Tensor<B, 3> {
        let [batch, heads, head_dim] = x.dims();
        let half = head_dim / 2;

        let sin = self
            .rope_sin
            .clone()
            .slice([time..time + 1, 0..half])
            .reshape([1, 1, half]);
        let cos = self
            .rope_cos
            .clone()
            .slice([time..time + 1, 0..half])
            .reshape([1, 1, half]);

        let first = x.clone().slice([0..batch, 0..heads, 0..half]);
        let second = x.slice([0..batch, 0..heads, half..head_dim]);
        Tensor::cat(
            vec![
                first.clone() * cos.clone() - second.clone() * sin.clone(),
                second * cos + first * sin,
            ],
            2,
        )
    }

    pub fn forward(&self, x: Tensor<B, 2>, time: usize, cache: &mut KvCache<B>) -> Tensor<B, 2> {
        let [batch, _] = x.dims();
        let (heads, kv_heads, head_dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        assert!(
            time < self.max_seq_length,
            "step {time} is past the {} the cache and rope tables hold; \
             the carry must be cycled first",
            self.max_seq_length
        );

        let query = x
            .clone()
            .matmul(self.query_proj.clone())
            .reshape([batch, heads, head_dim]);
        let key = x
            .clone()
            .matmul(self.key_proj.clone())
            .reshape([batch, kv_heads, head_dim]);
        let value = x
            .matmul(self.value_proj.clone())
            .reshape([batch, kv_heads, head_dim]);

        let query = self.rope(query, time);
        let key = self.rope(key, time);

        // flex copies the destination buffer on every `slice_assign` regardless
        // of how many handles it has (`slice_write_impl` clones inside
        // `to_contiguous`, so its `Arc::make_mut` never sees a lone reference),
        // so there is nothing to win by moving the cache out of `cache` first.
        let slot = [0..batch, 0..kv_heads, time..time + 1, 0..head_dim];
        cache.key = cache
            .key
            .clone()
            .slice_assign(slot.clone(), key.unsqueeze_dim(2));
        cache.value = cache
            .value
            .clone()
            .slice_assign(slot, value.unsqueeze_dim(2));

        // `attention` wants `[batch, heads, seq, head_dim]` with one head axis
        // shared by q/k/v, so grouped-query attention rides in on the query
        // length axis: a decode step has seq_q = 1, leaving that slot free for
        // the `groups` query heads that share a kv head. the cache is stored in
        // that layout, which matters: `attention` starts by forcing its inputs
        // contiguous, so a permuted view would copy the whole cache every layer
        // every step.
        let groups = heads / kv_heads;

        // hand over the whole buffer every step — one fixed kernel shape — and
        // say which slots are live with a mask. `is_causal` cannot stand in for
        // it: that flag reads the query's position off `seq_q`, which here
        // carries the groups, and anchors the query to the end of the keys.
        // on the last step of an episode every slot is written and the mask
        // drops away.
        //
        // broadcasting the mask over batch and kv heads stays zero-copy as long
        // as its trailing two dims match `[groups, max_seq_length]` exactly.
        let mask = (time + 1 < self.max_seq_length).then(|| {
            self.causal_mask
                .clone()
                .slice([time..time + 1, 0..self.max_seq_length])
                .reshape([1, 1, 1, self.max_seq_length])
                .expand([1, 1, groups, self.max_seq_length])
        });

        // the default scale is already 1/sqrt(head_dim); passing it explicitly
        // would push cubecl off its flash kernel onto the unfused fallback.
        let context = attention(
            query.reshape([batch, kv_heads, groups, head_dim]),
            cache.key.clone(),
            cache.value.clone(),
            mask,
            None,
            AttentionModuleOptions::default(),
        ); // [batch, kv, groups, head_dim]
        context
            .reshape([batch, heads * head_dim])
            .matmul(self.out_proj.clone())
    }
}

pub struct TransformerLayer<B: Backend> {
    /// `history_norm` + attention, absent when the layer has no history.
    pub history: Option<(RmsNorm<B>, Attention<B>)>,
    pub ffn_norm: RmsNorm<B>,
    pub ffn: FeedForward<B>,
}

/// Per-layer kv caches; `None` for layers without history.
pub struct Carry<B: Backend> {
    pub caches: Vec<Option<KvCache<B>>>,
}

/// The actor half of the jax `TransformerActorCritic`; the critic is trainer-
/// only, so the value head is not loaded and never runs.
pub struct TransformerActor<B: Backend> {
    pub obs_encoder: GridCnnEncoder<B>,
    pub reward_encoder: Linear<B>,
    pub action_embedder: Embedder<B>,
    pub layers: Vec<TransformerLayer<B>>,
    pub output_norm: RmsNorm<B>,
    pub max_seq_length: usize,
    pub action_dim: usize,
    pub device: B::Device,
}

impl<B: Backend> TransformerActor<B> {
    pub fn init_carry(&self, batch: usize) -> Carry<B> {
        Carry {
            caches: self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .history
                        .as_ref()
                        .map(|(_, attention)| attention.init_cache(batch, &self.device))
                })
                .collect(),
        }
    }

    /// One decode step for all agents at a shared `time`, returning normalized
    /// log-probs over actions with illegal actions masked out — exactly
    /// `distrax.Categorical(logits=masked_logits).logits`.
    pub fn step(
        &self,
        obs: ArrayView4<'_, u16>,
        reward: &[f32],
        last_action: &[u16],
        action_mask: ArrayView2<'_, bool>,
        time: usize,
        carry: &mut Carry<B>,
    ) -> Tensor<B, 2> {
        let batch = reward.len();
        let device = &self.device;

        let reward =
            Tensor::<B, 2>::from_data(TensorData::new(reward.to_vec(), [batch, 1]), device);
        let mut x = self.obs_encoder.forward(obs, device)
            + self.reward_encoder.forward(reward)
            + self.action_embedder.encode(last_action, device);

        for (layer, cache) in self.layers.iter().zip(&mut carry.caches) {
            if let Some((history_norm, attention)) = &layer.history {
                let cache = cache.as_mut().expect("carry built by init_carry");
                x = x.clone() + attention.forward(history_norm.forward(x), time, cache);
            }

            x = x.clone() + layer.ffn.forward(layer.ffn_norm.forward(x));
        }

        let x = self.output_norm.forward(x);

        let logits = self.action_embedder.decode(x);
        let illegal: Vec<bool> = action_mask.iter().map(|&legal| !legal).collect();
        let illegal = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(illegal, [batch, self.action_dim]),
            device,
        );
        log_softmax(logits.mask_fill(illegal, f32::MIN), 1)
    }
}
