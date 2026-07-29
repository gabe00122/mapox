use macroquad::prelude::*;

pub fn window_conf() -> Conf {
    Conf {
        window_title: "mapox".to_owned(),
        window_width: 800,
        window_height: 600,
        high_dpi: true,
        ..Default::default()
    }
}

pub fn open_window() {
    macroquad::Window::from_config(window_conf(), run());
}

pub async fn run() {
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
