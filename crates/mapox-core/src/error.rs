use std::io;

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
    #[error("cannot launch ffmpeg: {source}")]
    FfmpegSpawn {
        #[source]
        source: io::Error,
    },
    #[error("render state does not match render settings")]
    RenderSettingsMismatch,

    #[error(
        "observation shape mismatch: task {task:?} is {width}x{height}, but task {expected_task:?} is {expected_width}x{expected_height}"
    )]
    ObservationShapeMismatch {
        task: String,
        expected_task: String,
        width: i32,
        height: i32,
        expected_width: i32,
        expected_height: i32,
    },
}

pub type MapoxResult<T> = Result<T, MapoxError>;
