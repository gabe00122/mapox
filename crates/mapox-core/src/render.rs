//! Windowed rendering, shared by the native, python, and wasm entry points.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use macroquad::prelude::*;

/// Set once [`open_window`] starts a window. miniquad initializes global
/// display state that it never tears back down, so a second window in the same
/// process panics deep inside the backend; we refuse it up front instead.
static WINDOW_OPENED: AtomicBool = AtomicBool::new(false);

/// Reasons [`open_window`] can refuse to open a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowError {
    /// A window was already opened in this process.
    AlreadyOpened,
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowError::AlreadyOpened => f.write_str(
                "a window was already opened in this process; \
                 the graphics backend supports only one per process",
            ),
        }
    }
}

impl std::error::Error for WindowError {}

/// Window settings for the viewer.
pub fn window_conf() -> Conf {
    Conf {
        window_title: "mapox".to_owned(),
        window_width: 800,
        window_height: 600,
        high_dpi: true,
        ..Default::default()
    }
}

/// Opens the viewer window.
///
/// On desktop this blocks until the window closes and then returns, so callers
/// (including python) get control back. On wasm it hands off to the browser's
/// event loop and returns immediately.
///
/// Only one window may be opened per process; see [`WindowError`].
pub fn open_window() -> Result<(), WindowError> {
    if WINDOW_OPENED.swap(true, Ordering::SeqCst) {
        return Err(WindowError::AlreadyOpened);
    }

    macroquad::Window::from_config(window_conf(), run());
    Ok(())
}

/// Draws a red square in the middle of the window until it's closed or escape
/// is pressed.
pub async fn run() {
    // take over the close button so the loop below can break and unwind
    // normally; otherwise the backend tears the window down underneath us and
    // the caller never gets its stack back.
    prevent_quit();

    while !is_quit_requested() && !is_key_pressed(KeyCode::Escape) {
        clear_background(BLACK);

        let size = 200.0;
        draw_rectangle(
            (screen_width() - size) / 2.0,
            (screen_height() - size) / 2.0,
            size,
            size,
            RED,
        );

        next_frame().await
    }
}
