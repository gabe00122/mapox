//! Native, headless video recording. Inactive steps never touch the renderer or ffmpeg.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
};

use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    error::{MapoxError, MapoxResult},
    make::{EnvConfig, make},
    render::{
        env::{GridRenderSettings, GridRenderState},
        rgb::RgbRenderer,
    },
    spec::{ActionSpec, ObservationSpec},
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};

/// ffmpeg is spawned from PATH: the executable is deliberately not configurable.
const FFMPEG: &str = "ffmpeg";

/// One post-step frame per recorded step. FPS controls playback, not training speed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VideoConfig {
    pub env: EnvConfig,
    /// Use a separate directory for each run/worker; existing videos are never overwritten.
    pub output_dir: PathBuf,
    /// Number of step calls (frames) in each clip.
    pub record_steps: u64,
    /// Start-to-start spacing, at least `record_steps`. None records only one clip.
    pub interval_steps: Option<u64>,
    /// Zero-based step index of the first frame: 0 records immediately after the first step.
    pub start_step: u64,
    pub fps: u32,
    /// Output pixels; both dimensions must be positive and even for H.264/yuv420p.
    pub width: u32,
    pub height: u32,
    /// x264 CRF quality: 0 (lossless) to 51; higher compresses more. 23 is the default.
    pub crf: u8,
}

impl VideoConfig {
    fn validate(&self) -> MapoxResult<()> {
        if self.record_steps == 0
            || self
                .interval_steps
                .is_some_and(|interval| interval < self.record_steps)
            || self.fps == 0
            || self.fps > i32::MAX as u32
            || self.width == 0
            || self.height == 0
            || self.width % 2 != 0
            || self.height % 2 != 0
            || self.width > i32::MAX as u32
            || self.height > i32::MAX as u32
            || self.crf > 51
            || (self.width as usize)
                .checked_mul(self.height as usize)
                .and_then(|n| n.checked_mul(3))
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(MapoxError::InvalidConfig {
                reason: "video requires nonzero record_steps/fps, an interval >= record_steps, crf <= 51, and positive even dimensions within ffmpeg's limits".into()
            });
        }
        Ok(())
    }
}

/// Wrap outside a vector/multitask environment to record its selected render view,
/// not one encoder per training instance. Observations, rewards, metrics and resets
/// are forwarded unchanged. Resets neither add frames nor restart the schedule.
///
/// Each active step synchronously pipes one RGB frame to ffmpeg (bounded memory and
/// backpressure). Encoding/finalization can slow recorded steps, never inactive ones.
/// A recording failure disables future recording and is logged; the environment
/// continues stepping normally. Dropping the wrapper finalizes the current partial
/// clip best-effort. No display server or GPU is required.
pub struct VideoWrapper {
    inner: Box<dyn Environment>,
    config: VideoConfig,
    steps: u64,
    next_start: Option<u64>,
    recording: Option<Recording>,
}

impl VideoWrapper {
    /// Validates configuration only. No rendering, filesystem access or process
    /// creation happens until the first scheduled recording step.
    pub fn new(config: &VideoConfig, length: usize) -> MapoxResult<Self> {
        config.validate()?;

        let inner = make(&config.env, length)?;

        Ok(Self::with_inner(config, inner))
    }

    /// Wraps an already-built env. `new` is the supported entry point; tests use
    /// this to observe the wrapped env's render calls.
    fn with_inner(config: &VideoConfig, inner: Box<dyn Environment>) -> Self {
        Self {
            inner,
            next_start: Some(config.start_step),
            config: config.clone(),
            steps: 0,
            recording: None,
        }
    }

    fn finish_clip(&mut self) -> MapoxResult<()> {
        match self.recording.take() {
            Some(recording) => recording.encoder.finish(),
            None => Ok(()),
        }
    }

    fn record_step(&mut self, step: u64) -> MapoxResult<()> {
        if self.next_start == Some(step) {
            self.next_start = self
                .config
                .interval_steps
                .and_then(|interval| step.checked_add(interval));
            self.recording = Some(Recording::new(&*self.inner, &self.config, step)?);
        }
        if let Some(recording) = &mut self.recording {
            recording.capture(&*self.inner, &self.config)?;
            if recording.frames == self.config.record_steps {
                self.finish_clip()?;
            }
        }
        Ok(())
    }
}

impl Environment for VideoWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.inner.reset(seed, timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.inner.step(actions, timestep);
        let step = self.steps;
        self.steps = self.steps.saturating_add(1);
        if self.recording.is_none() && self.next_start != Some(step) {
            return;
        }
        if let Err(error) = self.record_step(step) {
            log::error!("video recording disabled at step {step}: {error}");
            self.recording = None;
            self.next_start = None;
        }
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.inner.observation_spec()
    }
    fn action_spec(&self) -> ActionSpec {
        self.inner.action_spec()
    }
    fn num_agents(&self) -> usize {
        self.inner.num_agents()
    }
    fn obs_vocab(&self) -> &Vocabulary {
        self.inner.obs_vocab()
    }
    fn action_vocab(&self) -> &Vocabulary {
        self.inner.action_vocab()
    }
    fn get_render_settings(&self) -> GridRenderSettings {
        self.inner.get_render_settings()
    }
    fn render_state_into(&self, state: &mut GridRenderState) {
        self.inner.render_state_into(state);
    }
    fn num_tasks(&self) -> usize {
        self.inner.num_tasks()
    }
    fn consume_metrics(&mut self) -> serde_json::Value {
        self.inner.consume_metrics()
    }

    fn set_enjoy_mode(&mut self, task_num: Option<usize>) {
        self.inner.set_enjoy_mode(task_num);
        // Switching tasks can change the map dimensions and vocabulary mid-clip.
        if let Some(recording) = &mut self.recording {
            recording.renderer = None;
        }
    }
}

impl Drop for VideoWrapper {
    fn drop(&mut self) {
        if let Err(error) = self.finish_clip() {
            log::error!("failed to finalize video: {error}");
        }
    }
}

struct Recording {
    encoder: Encoder,
    renderer: Option<RgbRenderer>,
    state: GridRenderState,
    frames: u64,
}

impl Recording {
    fn new(env: &dyn Environment, config: &VideoConfig, step: u64) -> MapoxResult<Self> {
        let renderer = RgbRenderer::new(&env.get_render_settings(), config.width, config.height)?;
        fs::create_dir_all(&config.output_dir)?;
        let path = config.output_dir.join(format!("video-{step:012}.mp4"));
        // Pass an exclusively created file as stdout: neither ffmpeg nor another
        // wrapper can truncate an existing recording, even across processes.
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let child = Command::new(FFMPEG)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
            ])
            .arg(format!("{}x{}", config.width, config.height))
            .arg("-framerate")
            .arg(config.fps.to_string())
            // No B-frame reordering: even partial fragments start at t=0,
            // without the decoder delay shifting their presentation timestamps.
            .args([
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf",
            ])
            .arg(config.crf.to_string())
            .args([
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "mp4",
                "-movflags",
                "+frag_keyframe+empty_moov",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::from(output))
            // Never leave an undrained stderr pipe capable of blocking the encoder.
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(source) => {
                let _ = fs::remove_file(&path);
                return Err(MapoxError::FfmpegSpawn { source });
            }
        };
        let stdin = child.stdin.take();
        Ok(Self {
            encoder: Encoder {
                child,
                stdin,
                reaped: false,
            },
            renderer: Some(renderer),
            state: GridRenderState::default(),
            frames: 0,
        })
    }

    fn capture(&mut self, env: &dyn Environment, config: &VideoConfig) -> MapoxResult<()> {
        if self.renderer.is_none() {
            self.renderer = Some(RgbRenderer::new(
                &env.get_render_settings(),
                config.width,
                config.height,
            )?);
        }
        env.render_state_into(&mut self.state);
        let pixels = self.renderer.as_mut().unwrap().render(&self.state)?;
        self.encoder.stdin.as_mut().unwrap().write_all(pixels)?;
        self.frames += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    reaped: bool,
}

impl Encoder {
    fn finish(mut self) -> MapoxResult<()> {
        // EOF flushes delayed frames and the container trailer before waiting.
        self.stdin.take();
        let status = self.child.wait()?;
        self.reaped = true;
        if !status.success() {
            return Err(MapoxError::FfmpegExited {
                status: status.to_string(),
            });
        }
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if !self.reaped {
            self.stdin.take();
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
