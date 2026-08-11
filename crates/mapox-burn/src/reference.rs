//! Reads the `*.io.safetensors` reference files the export script writes
//! (synthetic inputs plus the f32 jax model's outputs) and replays them
//! against a loaded model, so a fresh export can be checked end to end. The
//! reference file's `value` tensor is ignored: there is no value head here.

use burn::tensor::backend::Backend;
use ndarray::{Array2, Array4};
use safetensors::{Dtype, SafeTensors};

use crate::loader::LoadError;
use crate::model::{Carry, TransformerActor};

pub struct ReferenceSteps {
    pub steps: usize,
    pub num_agents: usize,
    pub action_dim: usize,
    /// `(steps, agents, w, h, channels)` flattened per step below.
    pub obs: Vec<Array4<u16>>,
    pub reward: Vec<Vec<f32>>,
    pub last_action: Vec<Vec<u16>>,
    pub action_mask: Vec<Array2<bool>>,
    pub log_probs: Vec<Vec<f32>>,
}

fn tensor_data<T: Copy>(
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: Dtype,
    from_bytes: impl Fn(&[u8]) -> T,
    width: usize,
) -> Result<(Vec<T>, Vec<usize>), LoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| format!("reference file is missing {name:?}"))?;
    if view.dtype() != dtype {
        return Err(format!("{name}: expected {dtype:?}, got {:?}", view.dtype()).into());
    }
    let data = view.data().chunks_exact(width).map(from_bytes).collect();
    Ok((data, view.shape().to_vec()))
}

pub fn load_reference(path: impl AsRef<std::path::Path>) -> Result<ReferenceSteps, LoadError> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("reading {}: {e}", path.as_ref().display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let f32s = |name: &str| {
        tensor_data(
            &tensors,
            name,
            Dtype::F32,
            |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            4,
        )
    };
    let u16s = |name: &str| {
        tensor_data(
            &tensors,
            name,
            Dtype::U16,
            |b| u16::from_le_bytes([b[0], b[1]]),
            2,
        )
    };
    let u8s = |name: &str| tensor_data(&tensors, name, Dtype::U8, |b| b[0], 1);

    let (obs, obs_shape) = u16s("obs")?;
    let [steps, agents, w, h, c] = obs_shape
        .clone()
        .try_into()
        .map_err(|_| format!("obs shape {obs_shape:?}, expected 5 dims"))?;
    let (reward, _) = f32s("reward")?;
    let (last_action, _) = u16s("last_action")?;
    let (mask, mask_shape) = u8s("action_mask")?;
    let action_dim = mask_shape[2];
    let (log_probs, _) = f32s("log_probs")?;

    let per_obs = agents * w * h * c;
    let per_mask = agents * action_dim;
    Ok(ReferenceSteps {
        steps,
        num_agents: agents,
        action_dim,
        obs: (0..steps)
            .map(|t| {
                Array4::from_shape_vec(
                    (agents, w, h, c),
                    obs[t * per_obs..(t + 1) * per_obs].to_vec(),
                )
                .expect("shape matches the slice")
            })
            .collect(),
        reward: reward.chunks(agents).map(<[f32]>::to_vec).collect(),
        last_action: last_action.chunks(agents).map(<[u16]>::to_vec).collect(),
        action_mask: (0..steps)
            .map(|t| {
                Array2::from_shape_vec(
                    (agents, action_dim),
                    mask[t * per_mask..(t + 1) * per_mask]
                        .iter()
                        .map(|&m| m != 0)
                        .collect(),
                )
                .expect("shape matches the slice")
            })
            .collect(),
        log_probs: log_probs.chunks(per_mask).map(<[f32]>::to_vec).collect(),
    })
}

/// Worst deviations from the reference over all steps.
#[derive(Debug, Default)]
pub struct ParityReport {
    /// Max abs diff over action *probabilities* (compares masked entries
    /// sanely: both sides are ~0 there).
    pub max_prob_diff: f32,
    /// Max abs diff over log-probs of *legal* actions only (masked entries
    /// sit near f32::MIN where absolute comparison is meaningless).
    pub max_legal_log_prob_diff: f32,
}

pub fn replay<B: Backend>(
    model: &TransformerActor<B>,
    reference: &ReferenceSteps,
) -> Result<ParityReport, LoadError> {
    let mut carry: Carry<B> = model.init_carry(reference.num_agents);
    let mut report = ParityReport::default();

    for t in 0..reference.steps {
        let log_probs = model
            .step(
                reference.obs[t].view(),
                &reference.reward[t],
                &reference.last_action[t],
                reference.action_mask[t].view(),
                t,
                &mut carry,
            )
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| format!("reading log probs: {e:?}"))?;

        for (i, (&ours, &expected)) in log_probs.iter().zip(&reference.log_probs[t]).enumerate() {
            let prob_diff = (ours.exp() - expected.exp()).abs();
            report.max_prob_diff = report.max_prob_diff.max(prob_diff);
            let legal = reference.action_mask[t]
                .as_slice()
                .expect("standard layout")[i];
            if legal {
                report.max_legal_log_prob_diff =
                    report.max_legal_log_prob_diff.max((ours - expected).abs());
            }
        }
    }

    Ok(report)
}
