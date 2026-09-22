use super::*;
use crate::{
    envs::snake::{Snake, SnakeConfig},
    timestep::TimeStepBuffers,
};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mapox-video-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn snake_config() -> SnakeConfig {
    SnakeConfig {
        num_agents: 1,
        width: 8,
        height: 8,
        ..Default::default()
    }
}

fn snake() -> Snake {
    Snake::new(&snake_config(), 64)
}

struct ObservedEnv {
    inner: Snake,
    settings_calls: Arc<AtomicUsize>,
    render_calls: Arc<AtomicUsize>,
}
impl Environment for ObservedEnv {
    fn reset(&mut self, seed: u64, ts: &mut TimeStepMut) {
        self.inner.reset(seed, ts);
    }
    fn step(&mut self, actions: &[VocabId], ts: &mut TimeStepMut) {
        self.inner.step(actions, ts);
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
        self.settings_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.get_render_settings()
    }
    fn render_state_into(&self, state: &mut GridRenderState) {
        self.render_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.render_state_into(state);
    }
    fn num_tasks(&self) -> usize {
        self.inner.num_tasks()
    }
    fn consume_metrics(&mut self) -> serde_json::Value {
        self.inner.consume_metrics()
    }
}

/// The python `RustVideoConfig` defaults, wrapping the same snake board [`snake`] builds.
fn video_config() -> VideoConfig {
    VideoConfig {
        env: EnvConfig::RustSnake(Box::new(snake_config())),
        output_dir: PathBuf::from("videos"),
        record_steps: 256,
        interval_steps: Some(10_000),
        start_step: 0,
        fps: 30,
        width: 640,
        height: 480,
        crf: 23,
    }
}

fn observed(config: VideoConfig) -> (VideoWrapper, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let settings = Arc::new(AtomicUsize::new(0));
    let renders = Arc::new(AtomicUsize::new(0));
    let env = ObservedEnv {
        inner: snake(),
        settings_calls: settings.clone(),
        render_calls: renders.clone(),
    };
    (
        VideoWrapper::with_inner(&config, Box::new(env)),
        settings,
        renders,
    )
}

#[test]
fn inactive_and_failed_recording_preserve_training_without_rendering_or_retries() {
    let dir = TestDir::new();
    // a file sitting where the clip directory belongs: creating the directory
    // fails deterministically, without depending on ffmpeg missing from PATH
    let blocker = dir.0.join("blocker");
    fs::write(&blocker, b"").unwrap();
    let output_dir = blocker.join("clips");
    let (mut video, settings, renders) = observed(VideoConfig {
        output_dir: output_dir.clone(),
        start_step: 3,
        record_steps: 2,
        interval_steps: Some(5),
        width: 96,
        height: 96,
        ..video_config()
    });
    let mut plain = snake();
    let mut actual = TimeStepBuffers::new(&video);
    let mut expected = TimeStepBuffers::new(&plain);
    video.reset(7, &mut actual.view_mut());
    plain.reset(7, &mut expected.view_mut());
    assert!(!output_dir.exists());
    assert_eq!(settings.load(Ordering::Relaxed), 0);
    for step in 0..12 {
        if step == 2 {
            video.reset(9, &mut actual.view_mut());
            plain.reset(9, &mut expected.view_mut());
        }
        video.step(&[0], &mut actual.view_mut());
        plain.step(&[0], &mut expected.view_mut());
        assert_eq!(actual.obs, expected.obs);
        assert_eq!(actual.reward, expected.reward);
        assert_eq!(actual.terminated, expected.terminated);
        assert_eq!(actual.time, expected.time);
        assert_eq!(actual.last_action, expected.last_action);
        assert_eq!(actual.action_mask, expected.action_mask);
        assert_eq!(actual.task_ids, expected.task_ids);
        if step < 3 {
            assert!(!output_dir.exists());
            assert_eq!(settings.load(Ordering::Relaxed), 0);
        } else {
            // recording is scheduled and the renderer is set up once, but the
            // clip directory cannot be created; the env must keep stepping
            // unaffected, and recording must not be retried
            assert_eq!(settings.load(Ordering::Relaxed), 1);
        }
    }
    assert_eq!(renders.load(Ordering::Relaxed), 0);
    assert_eq!(video.consume_metrics(), plain.consume_metrics());
    assert!(!output_dir.exists());
}

#[test]
fn rejects_unencodable_dimensions_and_overlapping_windows_before_io() {
    for config in [
        VideoConfig {
            width: 0,
            ..video_config()
        },
        VideoConfig {
            height: 15,
            ..video_config()
        },
        VideoConfig {
            fps: 0,
            ..video_config()
        },
        VideoConfig {
            record_steps: 0,
            ..video_config()
        },
        VideoConfig {
            record_steps: 10,
            interval_steps: Some(9),
            ..video_config()
        },
        VideoConfig {
            crf: 52,
            ..video_config()
        },
    ] {
        assert!(
            matches!(VideoWrapper::new(&config, 64), Err(MapoxError::InvalidConfig { .. }))
        );
    }
}

fn probe(path: &Path) -> serde_json::Value {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["streams"][0].clone()
}

#[test]
#[ignore = "requires ffmpeg with libx264 and ffprobe on PATH"]
fn scheduled_clips_cross_resets_and_flush_partial_video() {
    let dir = TestDir::new();
    let (mut video, _, renders) = observed(VideoConfig {
        output_dir: dir.0.clone(),
        start_step: 2,
        record_steps: 3,
        interval_steps: Some(5),
        fps: 12,
        width: 128,
        height: 96,
        ..video_config()
    });
    let mut ts = TimeStepBuffers::new(&video);
    video.reset(0, &mut ts.view_mut());
    for step in 0..9 {
        if step == 3 {
            video.reset(1, &mut ts.view_mut());
        }
        video.step(&[0], &mut ts.view_mut());
    }
    // Recording windows: steps 2..5 and 7..9, one render each.
    assert_eq!(renders.load(Ordering::Relaxed), 5);
    // Dropping finalizes both partial clips before probing.
    drop(video);
    for (start, frames) in [(2, 3), (7, 2)] {
        let path = dir.0.join(format!("video-{start:012}.mp4"));
        let stream = probe(&path);
        assert_eq!(stream["nb_read_frames"], frames.to_string());
        assert_eq!(stream["width"], 128);
        assert_eq!(stream["height"], 96);
        // r_frame_rate is ffprobe's guessed base rate (24 for a two-frame
        // 12fps clip); average rate and duration describe actual playback.
        assert_eq!(stream["avg_frame_rate"], "12/1");
        assert_eq!(stream["start_time"], "0.000000");
        let duration: f64 = stream["duration"].as_str().unwrap().parse().unwrap();
        assert!((duration - f64::from(frames) / 12.0).abs() < 0.001);
    }
}

#[test]
#[ignore = "requires ffmpeg with libx264 and ffprobe on PATH"]
fn drop_flushes_and_existing_videos_are_not_overwritten() {
    let dir = TestDir::new();
    let config = VideoConfig {
        output_dir: dir.0.clone(),
        record_steps: 10,
        interval_steps: None,
        width: 96,
        height: 96,
        ..video_config()
    };
    let (mut video, _, _) = observed(config.clone());
    let mut ts = TimeStepBuffers::new(&video);
    video.reset(0, &mut ts.view_mut());
    video.step(&[0], &mut ts.view_mut());
    drop(video);
    let path = dir.0.join("video-000000000000.mp4");
    assert_eq!(probe(&path)["nb_read_frames"], "1");
    let original = fs::read(&path).unwrap();
    let (mut collision, _, _) = observed(config);
    collision.reset(0, &mut ts.view_mut());
    collision.step(&[0], &mut ts.view_mut());
    // The already-taken file name refuses the new recording (create_new), so
    // the pre-existing video stays intact.
    assert_eq!(fs::read(path).unwrap(), original);
}
