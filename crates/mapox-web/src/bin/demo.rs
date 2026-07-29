fn main() {
    if let Err(err) = mapox_core::render::open_window() {
        eprintln!("mapox: {err}");
        std::process::exit(1);
    }
}
