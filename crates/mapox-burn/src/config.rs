//! The dimensions the model is built from, parsed out of the JSON the export
//! script embeds in safetensors metadata. Any other key in that JSON is
//! ignored.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub obs_encoder: GridCnnConfig,
    pub hidden_features: usize,
    pub layer: LayerConfig,
    pub num_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GridCnnConfig {
    pub kernels: Vec<[usize; 2]>,
    pub strides: Vec<[usize; 2]>,
    pub channels: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerConfig {
    pub feed_forward: FeedForwardConfig,
    #[serde(default)]
    pub history: Option<AttentionConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeedForwardConfig {
    pub size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rope_max_wavelength")]
    pub rope_max_wavelength: f64,
}

/// Everything the export script recorded about the policy besides weights.
#[derive(Debug, Clone)]
pub struct PolicyMetadata {
    pub model: ModelConfig,
    /// `(view_width, view_height, channels)` of one agent's observation.
    pub obs_shape: [usize; 3],
    /// Per-channel vocabulary size; the cnn one-hot encodes each channel.
    pub obs_max_value: Vec<usize>,
    pub action_dim: usize,
    pub max_seq_length: usize,
    /// The training run's env config JSON, deserializable into
    /// `mapox_core::make::EnvConfig`.
    pub env_config: Option<String>,
    /// `<run>@<step>` or `fixture:<name>`.
    pub source: String,
}

fn default_rope_max_wavelength() -> f64 {
    10_000.0
}
