//! The egui theme, matched to the personal site the web demo is embedded in
//! (`personal-site/src/app.css`): dark only, square corners, flat surfaces,
//! and selected things highlighted in the tileset yellow like a selected cell.

use egui::{Color32, CornerRadius, Shadow, Stroke, Theme, Visuals, style::WidgetVisuals};

// The site's palette, by its CSS variable names.
pub const BG: Color32 = Color32::from_rgb(0x10, 0x12, 0x1a);
pub const BG_DEEP: Color32 = Color32::from_rgb(0x0b, 0x0c, 0x11);
pub const SURFACE: Color32 = Color32::from_rgb(0x1c, 0x1f, 0x27);
pub const FG: Color32 = Color32::from_rgb(0xe8, 0xec, 0xf4);
pub const MUTED: Color32 = Color32::from_rgb(0x8a, 0x93, 0xa4);
pub const BORDER: Color32 = Color32::from_rgb(0x22, 0x26, 0x34);
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x2c, 0x33, 0x49);
pub const ACCENT: Color32 = Color32::from_rgb(0xff, 0xd4, 0x3b);
pub const ACCENT_FG: Color32 = BG;

/// Gains, like a positive reward. The site has no green, so this one is the
/// viewer's own; losses use egui's `error_fg_color`.
pub const POSITIVE: Color32 = Color32::from_rgb(120, 200, 120);

/// Plain labels, halfway between [`FG`] and [`MUTED`], so `strong` text (the
/// section headings) still stands out at [`FG`] and `weak` text sits at
/// [`MUTED`] below them.
const BODY: Color32 = Color32::from_rgb(0xb9, 0xbf, 0xcc);

/// Install the theme for both of egui's themes and pin it to dark; the site
/// has no light mode, so neither does the viewer.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.all_styles_mut(|style| style.visuals = visuals());
}

fn visuals() -> Visuals {
    let widget = |fill: Color32, stroke: Stroke, text: Color32| WidgetVisuals {
        bg_fill: fill,
        weak_bg_fill: fill,
        bg_stroke: stroke,
        corner_radius: CornerRadius::ZERO,
        fg_stroke: Stroke::new(1.0, text),
        expansion: 0.0,
    };

    let mut visuals = Visuals::dark();
    visuals.weak_text_color = Some(MUTED);
    visuals.hyperlink_color = ACCENT;
    visuals.faint_bg_color = SURFACE;
    visuals.extreme_bg_color = BG_DEEP;
    visuals.text_edit_bg_color = Some(SURFACE);
    visuals.code_bg_color = SURFACE;

    visuals.panel_fill = BG;
    visuals.window_fill = SURFACE;
    visuals.window_stroke = Stroke::new(2.0, BORDER_STRONG);
    visuals.window_corner_radius = CornerRadius::ZERO;
    visuals.menu_corner_radius = CornerRadius::ZERO;
    visuals.window_shadow = Shadow::NONE;
    visuals.popup_shadow = Shadow::NONE;

    // buttons sit on the surface behind a strong border, which brightens on
    // hover; pressed or focused, it turns the accent like the site's focus ring
    visuals.widgets.noninteractive = widget(BG, Stroke::new(1.0, BORDER), BODY);
    visuals.widgets.inactive = widget(SURFACE, Stroke::new(1.0, BORDER_STRONG), FG);
    visuals.widgets.hovered = widget(SURFACE, Stroke::new(1.0, MUTED), FG);
    visuals.widgets.active = widget(SURFACE, Stroke::new(1.0, ACCENT), FG);
    visuals.widgets.open = widget(SURFACE, Stroke::new(1.0, BORDER_STRONG), FG);

    // selected toggles and selected text highlight like a selected cell
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT_FG);
    visuals.text_cursor.stroke = Stroke::new(2.0, ACCENT);

    visuals
}
