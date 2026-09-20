use std::{io, path::PathBuf};

use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum MapoxError {
    #[error("Mapox config was invalid: {reason}")]
    InvalidConfig { reason: String },

    #[error("ffmpeg exited with {status:?}; see stderr for details")]
    FfmpegExited { status: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("cannot launch ffmpeg {exe:?}: {source}")]
    FfmpegSpawn {
        exe: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("render state does not match render settings")]
    RenderSettingsMismatch,
}

pub type MapoxResult<T> = Result<T, MapoxError>;
