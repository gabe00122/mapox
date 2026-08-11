use mapox_burn::config::ModelConfig;

/// A full jaxrl config dump: the model reads four keys out of it and ignores
/// the rest, including the whole critic.
const FULL: &str = r#"{
  "obs_encoder": {"obs_type": "grid_cnn", "kernels": [[3,3],[3,3]], "strides": [[1,1],[1,1]], "channels": [32]},
  "hidden_features": 128,
  "layer": {
    "feed_forward": {"size": 256, "glu": true},
    "history": {"type": "attention", "num_heads": 4, "num_kv_heads": 2, "head_dim": 32,
                "sliding_window": null, "rope_max_wavelength": 10000.0, "use_qk_norm": false},
    "use_post_attn_norm": false,
    "use_post_ffw_norm": false
  },
  "num_layers": 2,
  "value_hidden_dim": 64,
  "value": {"type": "hl_gauss", "min": -10.0, "max": 10.0, "n_logits": 51, "sigma": 0.75},
  "activation": "gelu",
  "norm": "rms_norm",
  "kernel_init": "lecun_normal",
  "dtype": "bfloat16",
  "param_dtype": "float32"
}"#;

#[test]
fn reads_the_dimensions_out_of_a_full_config() {
    let parsed: ModelConfig = serde_json::from_str(FULL).unwrap();
    assert_eq!(parsed.num_layers, 2);
    assert_eq!(parsed.hidden_features, 128);
    assert_eq!(parsed.layer.feed_forward.size, 256);
    let attention = parsed.layer.history.expect("history is attention");
    assert_eq!(attention.num_heads, 4);
    assert_eq!(attention.head_dim, 32);
    assert_eq!(parsed.obs_encoder.channels, vec![32]);
}

/// A layer with no history parses; the model then skips attention entirely.
#[test]
fn history_is_optional() {
    let json = FULL.replace(
        r#""history": {"type": "attention", "num_heads": 4, "num_kv_heads": 2, "head_dim": 32,
                "sliding_window": null, "rope_max_wavelength": 10000.0, "use_qk_norm": false},"#,
        r#""history": null,"#,
    );
    let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
    assert!(parsed.layer.history.is_none());
}
