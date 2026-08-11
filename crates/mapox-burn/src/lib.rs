//! Burn inference for jaxrl-trained mapox policies.
//!
//! jaxrl's `scripts/export_burn_policy.py` writes a checkpoint to a single
//! safetensors bundle; [`load_policy`] rebuilds the actor-critic on any burn
//! backend and [`BurnPolicy`] drives it as a [`mapox_core::policy::Policy`],
//! so the native viewer can run a trained agent without python or jax:
//!
//! ```text
//! uv run scripts/export_burn_policy.py <run>            # in jaxrl
//! cargo run -p mapox-burn --example demo -- results/<run>/burn/policy.safetensors
//! ```

pub mod config;
pub mod loader;
pub mod model;
pub mod policy;
pub mod reference;

/// Re-exported so downstream crates name the backend (`mapox_burn::burn::
/// backend::NdArray`) without having to keep a burn version in sync with the
/// one these weights were built against.
pub use burn;

pub use loader::{LoadError, LoadedPolicy, load_policy, load_policy_bytes};
pub use policy::BurnPolicy;
