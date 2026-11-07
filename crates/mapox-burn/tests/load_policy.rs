use std::collections::HashMap;

use mapox_burn::burn::backend::{Flex, flex::FlexDevice};
use mapox_burn::load_policy_bytes;
use safetensors::{Dtype, tensor::TensorView};
use serde_json::json;

/// A small multitask export, with the task embedding optionally omitted as
/// it is in jaxrl runs trained with `task_embedding: false`.
fn bundle(task_embedding: Option<bool>, include_task_tensor: bool) -> Vec<u8> {
    let mut config = json!({
        "obs_encoder": {
            "kernels": [[1, 1]], "strides": [[1, 1]],
            "channels": [], "embedding_dim": 2
        },
        "hidden_features": 2,
        "layer": {"feed_forward": {"size": 2}},
        "num_layers": 0
    });
    if let Some(enabled) = task_embedding {
        config["task_embedding"] = json!(enabled);
    }
    let metadata = HashMap::from([
        ("format".into(), "mapox-burn-v1".into()),
        ("model_config".into(), config.to_string()),
        ("obs_shape".into(), "[1, 1, 1]".into()),
        ("obs_max_value".into(), "[2]".into()),
        ("action_dim".into(), "2".into()),
        ("num_tasks".into(), "2".into()),
        ("max_seq_length".into(), "4".into()),
        ("source".into(), "fixture:task_embedding".into()),
    ]);
    let mut shapes = vec![
        ("obs_encoder.embedding", vec![2, 2]),
        ("obs_encoder.layers.0.kernel", vec![1, 1, 2, 2]),
        ("obs_encoder.layers.0.bias", vec![2]),
        ("reward_encoder.kernel", vec![1, 2]),
        ("reward_encoder.bias", vec![2]),
        ("action_embedder.embedding_table", vec![2, 2]),
        ("output_norm.scale", vec![2]),
    ];
    if include_task_tensor {
        shapes.push(("task_embedder.embedding_table", vec![2, 2]));
    }
    let tensors: Vec<_> = shapes
        .into_iter()
        .map(|(name, shape)| {
            let data = vec![0_u8; shape.iter().product::<usize>() * 4];
            (name, shape, data)
        })
        .collect();
    let views = tensors.iter().map(|(name, shape, data)| {
        (
            *name,
            TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
        )
    });
    safetensors::serialize(views, Some(metadata)).unwrap()
}

#[test]
fn multitask_exports_load_with_and_without_task_embeddings() {
    for (setting, has_embedding) in [(None, true), (Some(true), true), (Some(false), false)] {
        let loaded = load_policy_bytes::<Flex>(&bundle(setting, has_embedding), &FlexDevice)
            .expect("valid multitask export loads");
        assert_eq!(loaded.model.task_embedder.is_some(), has_embedding);
    }
}

#[test]
fn missing_required_task_embeddings_are_still_rejected() {
    for setting in [None, Some(true)] {
        let error = match load_policy_bytes::<Flex>(&bundle(setting, false), &FlexDevice) {
            Ok(_) => panic!("an enabled task embedding needs its weights"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("task_embedder.embedding_table"));
    }
}
