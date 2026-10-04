use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eye_tracker_core::{CALIBRATION_SAMPLES_PER_TARGET, CalibrationOutcome};
use eyeos::{
    CalibrationProfile, ControlEngine, EngineEvent, EyeTracker, GazeSample, InputAction,
    InputController, InteractionMode, Point, SafetyState, TrackerConfig, TrackingState,
    config::AppConfig,
    display::{DisplayGeometry, physical_to_logical, primary_display_geometry},
    persistence::{ProfileStore, install_autostart},
    tracker::TrackerStatus,
    vision::{CameraStatus, EyeFeatures, ModelStatus, detect_camera_status, model_status},
};

const BLOB_SIZE: f32 = 80.0;
const PANEL_SIZE: f32 = 300.0;
const KEYBOARD_WIDTH: f32 = 720.0;
const KEYBOARD_HEIGHT: f32 = 340.0;
const OVERLAY_MARGIN: f32 = 16.0;

#[derive(Debug, Parser)]
#[command(
    name = "eyeos",
    about = "Offline, safety-first Windows desktop eye control"
)]
struct Cli {
    /// Open the caregiver calibration and settings screen.
    #[arg(long)]
    setup: bool,
    /// Open the isolated training environment.
    #[arg(long)]
    training: bool,
    /// Start EyeOS automatically after this Windows user signs in.
    #[arg(long)]
    install_autostart: bool,
    /// Remove locally stored settings and the encrypted calibration profile.
    #[arg(long)]
    reset_profile: bool,
    /// Exercise gaze dwell logic from the physical mouse position without injecting desktop input.
    #[arg(long, hide = true)]
    simulate_gaze: bool,
}

/// The normal launch surface is deliberately only the blob. Full-sized screens are available
/// only for caregiver setup/training or after an intentional gaze dwell on the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Overlay,
    Actions,
    Keyboard,
    Calibration,
    Training,
    Setup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlayAction {
    LeftClick,
    DoubleClick,
    RightClick,
    Drag,
    Scroll,
    Keyboard,
    Training,
    Calibrate,
    Pause,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyboardAction {
    Text(&'static str),
    Backspace,
    Enter,
    Phrase(usize),
    Back,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OverlayTarget {
    Blob,
    Action(OverlayAction),
    Key(KeyboardAction),
}

struct EyeOsApp {
    store: ProfileStore,
    config: AppConfig,
    engine: ControlEngine,
    input: InputController,
    page: Page,
    camera: CameraStatus,
    model: ModelStatus,
    screen_size: Point,
    started_at: Instant,
    simulate_gaze: bool,
    training_text: String,
    status_message: String,
    dwell_progress: f32,
    calibration: Option<CalibrationProfile>,
    tracker: Option<EyeTracker>,
    latest_features: Option<(EyeFeatures, u64)>,
    overlay_target: Option<OverlayTarget>,
    overlay_target_started_at: Option<u64>,
    overlay_cooldown_until: u64,
    detected_display: Option<DisplayGeometry>,
    gaze_preview: Option<Point>,
}

impl EyeOsApp {
    fn new(store: ProfileStore, mut config: AppConfig, page: Page, simulate_gaze: bool) -> Self {
        let detected_display = primary_display_geometry();
        if let Some(display) = &detected_display {
            if config.screen_width_mm.is_none() && config.screen_height_mm.is_none() {
                config.screen_width_mm = Some(display.size_mm.x);
                config.screen_height_mm = Some(display.size_mm.y);
                let _ = store.save_config(&config);
            }
        }
        let screen_size = primary_screen_size();
        let mut engine = ControlEngine::new(screen_size.x, screen_size.y);
        engine.dwell_ms = config.dwell_ms;
        engine.keyboard_dwell_ms = config.keyboard_dwell_ms;

        let mut tracker = EyeTracker::new(
            tracking_config(screen_size, &config),
            store.root().to_path_buf(),
        )
        .ok();
        let calibration = store.load_calibration().ok().flatten().filter(|profile| {
            tracker
                .as_mut()
                .is_some_and(|tracker| tracker.engine_mut().set_profile(profile.clone()).is_ok())
        });
        let started_at = tracker
            .as_ref()
            .map(|t| t.clock())
            .unwrap_or_else(Instant::now);
        let model = model_status();
        let mut app = Self {
            store,
            config,
            engine,
            input: InputController::default(),
            page,
            camera: detect_camera_status(),
            model,
            screen_size,
            started_at,
            simulate_gaze,
            training_text: String::new(),
            status_message: "Looking for the local eye tracker…".to_owned(),
            dwell_progress: 0.0,
            calibration,
            tracker,
            latest_features: None,
            overlay_target: None,
            overlay_target_started_at: None,
            overlay_cooldown_until: 0,
            detected_display,
            gaze_preview: None,
        };

        // A user who has a reviewed local model and a saved calibration should not need a
        // caregiver menu on every sign-in. Until both exist, the blob remains visibly paused.
        if app.simulate_gaze {
            // The simulator is deliberately dry-run only. It is a way to verify the complete
            // dwell/action route without allowing a development cursor to click the desktop.
            let events = app.engine.set_paused(false);
            app.process_events(events);
            app.status_message = "Developer gaze simulation — dry-run only.".to_owned();
        } else {
            if let Some(tracker) = app.tracker.as_mut() {
                if let Err(error) = tracker.start_camera(app.config.camera_index) {
                    app.status_message = format!("Eye tracker could not start: {error}");
                    return app;
                }
            } else {
                app.status_message = "Invalid tracker settings; review setup.".into();
            }
        }

        if !app.simulate_gaze
            && app.model == ModelStatus::Ready
            && app.has_validated_calibration()
            && app.config.live_input_confirmed
        {
            app.input.set_dry_run(false);
            let events = app.engine.set_paused(false);
            app.process_events(events);
        } else if !app.simulate_gaze {
            app.status_message = if !app.has_validated_calibration() {
                "Tracker starting — complete quick calibration and independent precision validation.".to_owned()
            } else {
                "Paused: caregiver confirmation is required before live input.".to_owned()
            };
        }
        if app.page == Page::Overlay && app.calibration.is_none() && !app.simulate_gaze {
            app.page = Page::Setup;
        }
        app
    }

    fn dispatch(&mut self, action: InputAction) {
        if let Err(error) = self.input.dispatch(action) {
            self.status_message = format!("Input was not sent: {error}");
        }
    }

    fn has_validated_calibration(&self) -> bool {
        self.calibration
            .as_ref()
            .is_some_and(|profile| profile.validation_passed)
            && self
                .tracker
                .as_ref()
                .and_then(|t| t.engine().profile())
                .is_some_and(|p| p.validation_passed)
    }

    fn process_events(&mut self, events: Vec<EngineEvent>) {
        for event in events {
            match event {
                EngineEvent::Action(action) => self.dispatch(action),
                EngineEvent::SafetyChanged(state) => {
                    self.status_message = match state {
                        SafetyState::Paused => {
                            "Paused — no desktop input is being sent.".to_owned()
                        }
                        SafetyState::Tracking => "Eye tracking is active.".to_owned(),
                        SafetyState::TrackingLost => {
                            "Tracking lost — any held drag was released.".to_owned()
                        }
                    };
                }
                EngineEvent::DwellProgress(progress) => self.dwell_progress = progress,
            }
        }
    }

    fn save_config(&mut self) {
        self.config.dwell_ms = self.engine.dwell_ms;
        self.config.keyboard_dwell_ms = self.engine.keyboard_dwell_ms;
        match self.store.save_config(&self.config) {
            Ok(()) => self.status_message = "Settings saved locally.".to_owned(),
            Err(error) => self.status_message = format!("Could not save settings: {error}"),
        }
    }

    fn set_page(&mut self, page: Page, context: &egui::Context) {
        self.page = page;
        self.clear_overlay_target();
        let fullscreen = matches!(page, Page::Setup | Page::Calibration);
        context.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        context.send_viewport_cmd(egui::ViewportCommand::Transparent(!fullscreen));
        if fullscreen {
            let screen = primary_ui_size();
            context.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::new(
                screen.x as f32,
                screen.y as f32,
            )));
            context.send_viewport_cmd(egui::ViewportCommand::OuterPosition(Pos2::ZERO));
            return;
        }
        let screen = physical_to_logical(self.screen_size, context.pixels_per_point());
        let (size, position) = match page {
            Page::Overlay => (
                Vec2::splat(BLOB_SIZE),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (screen.y as f32 - BLOB_SIZE - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Actions => (
                Vec2::splat(PANEL_SIZE),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (screen.y as f32 - PANEL_SIZE - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Keyboard => (
                Vec2::new(KEYBOARD_WIDTH, KEYBOARD_HEIGHT),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (screen.y as f32 - KEYBOARD_HEIGHT - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Calibration => (
                Vec2::new(self.screen_size.x as f32, self.screen_size.y as f32),
                Pos2::ZERO,
            ),
            Page::Training | Page::Setup => (Vec2::new(620.0, 620.0), Pos2::new(80.0, 80.0)),
        };
        context.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
        context.send_viewport_cmd(egui::ViewportCommand::OuterPosition(position));
    }

    fn toggle_tracking(&mut self) {
        if self.model != ModelStatus::Ready || !self.has_validated_calibration() {
            self.status_message =
                "Tracking cannot start until a local model and calibration are available."
                    .to_owned();
            return;
        }
        if !self.config.live_input_confirmed && !self.simulate_gaze {
            self.status_message =
                "Enable live desktop mouse control in this workspace first.".into();
            return;
        }
        self.input.set_dry_run(self.simulate_gaze);
        let pause = self.engine.safety != SafetyState::Paused;
        let events = self.engine.set_paused(pause);
        self.process_events(events);
    }

    fn select_mode(&mut self, mode: InteractionMode) {
        self.resume_tracking_after_intent();
        self.engine.set_mode(mode);
        self.status_message = format!("{} selected.", mode_label(mode));
    }

    fn resume_tracking_after_intent(&mut self) {
        if self.engine.safety == SafetyState::Tracking {
            return;
        }
        let can_resume = self.simulate_gaze
            || (self.model == ModelStatus::Ready
                && self.has_validated_calibration()
                && self.config.live_input_confirmed
                && !self.input.is_dry_run());
        if can_resume {
            let events = self.engine.set_paused(false);
            self.process_events(events);
        }
    }

    /// This is the single entry point for real tracker samples. A future tracker must call this
    /// with calibrated screen coordinates; raw webcam pixels are never treated as gaze.
    fn process_gaze_sample(&mut self, sample: GazeSample, context: &egui::Context) {
        if sample.confidence < self.engine.minimum_confidence {
            self.clear_overlay_target();
            let events = self.engine.update(sample);
            self.process_events(events);
            return;
        }

        match self.page {
            Page::Overlay => {
                if self.in_blob(sample.position) {
                    self.update_overlay_target(OverlayTarget::Blob, sample.timestamp_ms, context);
                } else {
                    self.clear_overlay_target();
                    let events = self.engine.update(sample);
                    self.process_events(events);
                }
            }
            Page::Actions => {
                if let Some(action) = self.action_at(sample.position) {
                    self.update_overlay_target(
                        OverlayTarget::Action(action),
                        sample.timestamp_ms,
                        context,
                    );
                } else {
                    self.clear_overlay_target();
                }
            }
            Page::Keyboard => {
                if let Some(action) = self.keyboard_action_at(sample.position) {
                    self.update_overlay_target(
                        OverlayTarget::Key(action),
                        sample.timestamp_ms,
                        context,
                    );
                } else {
                    self.clear_overlay_target();
                }
            }
            Page::Setup => {
                let events = self.engine.update(sample);
                self.process_events(events);
            }
            Page::Calibration | Page::Training => {}
        }
    }

    fn update_overlay_target(
        &mut self,
        target: OverlayTarget,
        timestamp_ms: u64,
        context: &egui::Context,
    ) {
        if timestamp_ms < self.overlay_cooldown_until {
            return;
        }
        if self.overlay_target.as_ref() != Some(&target) {
            self.overlay_target = Some(target);
            self.overlay_target_started_at = Some(timestamp_ms);
            self.dwell_progress = 0.0;
            return;
        }

        let started_at = self.overlay_target_started_at.unwrap_or(timestamp_ms);
        let dwell = match target {
            OverlayTarget::Blob => 800,
            OverlayTarget::Action(_) => self.engine.dwell_ms,
            OverlayTarget::Key(_) => self.engine.keyboard_dwell_ms,
        };
        let elapsed = timestamp_ms.saturating_sub(started_at);
        self.dwell_progress = (elapsed as f32 / dwell as f32).clamp(0.0, 1.0);
        if elapsed < dwell {
            return;
        }

        self.overlay_cooldown_until = timestamp_ms + self.engine.cooldown_ms;
        self.clear_overlay_target();
        match target {
            OverlayTarget::Blob => self.set_page(Page::Actions, context),
            OverlayTarget::Action(action) => self.activate_action(action, context),
            OverlayTarget::Key(action) => self.activate_keyboard_action(action, context),
        }
    }

    fn clear_overlay_target(&mut self) {
        self.overlay_target = None;
        self.overlay_target_started_at = None;
        self.dwell_progress = 0.0;
    }

    fn activate_action(&mut self, action: OverlayAction, context: &egui::Context) {
        match action {
            OverlayAction::LeftClick => self.select_mode(InteractionMode::Pointer),
            OverlayAction::DoubleClick => self.select_mode(InteractionMode::DoubleClick),
            OverlayAction::RightClick => self.select_mode(InteractionMode::RightClick),
            OverlayAction::Drag => self.select_mode(InteractionMode::DragReady),
            OverlayAction::Scroll => self.select_mode(InteractionMode::Scroll),
            OverlayAction::Keyboard => {
                self.select_mode(InteractionMode::Keyboard);
                self.set_page(Page::Keyboard, context);
                return;
            }
            OverlayAction::Training => {
                self.input.set_dry_run(true);
                self.set_page(Page::Training, context);
                return;
            }
            OverlayAction::Calibrate => {
                self.set_page(Page::Setup, context);
                return;
            }
            OverlayAction::Pause => {
                let pause = self.engine.safety == SafetyState::Tracking;
                let events = self.engine.set_paused(pause);
                self.process_events(events);
            }
        }
        self.set_page(Page::Overlay, context);
    }

    fn activate_keyboard_action(&mut self, action: KeyboardAction, context: &egui::Context) {
        match action {
            KeyboardAction::Text(value) => self.dispatch(InputAction::Text(value.to_owned())),
            KeyboardAction::Backspace => self.dispatch(InputAction::KeyChord {
                ctrl: false,
                shift: false,
                alt: false,
                virtual_key: 0x08,
            }),
            KeyboardAction::Enter => self.dispatch(InputAction::Text("\n".to_owned())),
            KeyboardAction::Phrase(index) => {
                if let Some(phrase) = self.config.phrase_cards.get(index) {
                    self.dispatch(InputAction::Text(phrase.clone()));
                }
            }
            KeyboardAction::Back => {
                self.select_mode(InteractionMode::Pointer);
                self.set_page(Page::Overlay, context);
            }
        }
    }

    fn in_blob(&self, point: Point) -> bool {
        let origin = self.overlay_origin(BLOB_SIZE);
        let center = Point::new(
            origin.x + f64::from(BLOB_SIZE / 2.0),
            origin.y + f64::from(BLOB_SIZE / 2.0),
        );
        point.distance_to(center) <= f64::from(BLOB_SIZE * 0.45)
    }

    fn action_at(&self, point: Point) -> Option<OverlayAction> {
        let origin = self.overlay_origin(PANEL_SIZE);
        let local_x = point.x - origin.x;
        let local_y = point.y - origin.y;
        if !(0.0..f64::from(PANEL_SIZE)).contains(&local_x)
            || !(0.0..f64::from(PANEL_SIZE)).contains(&local_y)
        {
            return None;
        }
        let column = (local_x / 100.0) as usize;
        let row = (local_y / 100.0) as usize;
        ACTIONS.get(row * 3 + column).copied()
    }

    fn keyboard_action_at(&self, point: Point) -> Option<KeyboardAction> {
        let origin = self.overlay_origin(KEYBOARD_HEIGHT);
        let x = point.x - origin.x;
        let y = point.y - origin.y;
        if !(0.0..f64::from(KEYBOARD_WIDTH)).contains(&x)
            || !(0.0..f64::from(KEYBOARD_HEIGHT)).contains(&y)
        {
            return None;
        }

        if y < 60.0 {
            return KEY_ROWS[0]
                .get((x / 72.0) as usize)
                .map(|key| KeyboardAction::Text(key));
        }
        if y < 120.0 {
            return KEY_ROWS[1]
                .get((x / 72.0) as usize)
                .map(|key| KeyboardAction::Text(key));
        }
        if y < 180.0 {
            return KEY_ROWS[2]
                .get((x / 72.0) as usize)
                .map(|key| KeyboardAction::Text(key));
        }
        if y < 260.0 {
            return if x < 216.0 {
                Some(KeyboardAction::Text(" "))
            } else if x < 360.0 {
                Some(KeyboardAction::Backspace)
            } else if x < 504.0 {
                Some(KeyboardAction::Enter)
            } else {
                Some(KeyboardAction::Back)
            };
        }
        if y < 340.0 {
            return if x < 240.0 {
                Some(KeyboardAction::Text("the "))
            } else if x < 480.0 {
                Some(KeyboardAction::Text("and "))
            } else if self.config.phrase_cards.is_empty() {
                Some(KeyboardAction::Text("thank you"))
            } else {
                Some(KeyboardAction::Phrase(0))
            };
        }
        None
    }

    fn overlay_origin(&self, height: f32) -> Point {
        Point::new(
            f64::from(OVERLAY_MARGIN),
            (self.screen_size.y - f64::from(height) - f64::from(OVERLAY_MARGIN)).max(0.0),
        )
    }

    fn render_overlay(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, Sense::click());
        let center = rect.center();
        let colour = match self.engine.safety {
            SafetyState::Paused => Color32::from_rgb(228, 170, 46),
            SafetyState::Tracking => Color32::from_rgb(52, 208, 131),
            SafetyState::TrackingLost => Color32::from_rgb(235, 88, 88),
        };
        let fill = Color32::from_rgba_unmultiplied(
            19,
            31,
            44,
            (self.config.overlay_opacity * 245.0) as u8,
        );
        ui.painter().circle_filled(center, 33.0, fill);
        ui.painter()
            .circle_stroke(center, 33.0, Stroke::new(4.0_f32, colour));
        ui.painter().circle_filled(center, 7.0, colour);
        if self.dwell_progress > 0.0 {
            ui.painter().circle_stroke(
                center,
                27.0,
                Stroke::new(3.0_f32, Color32::WHITE.linear_multiply(self.dwell_progress)),
            );
        }
        if response.clicked() {
            self.set_page(Page::Actions, context);
        }
        response.on_hover_text(self.status_message.clone());
    }

    fn render_actions(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        for (index, action) in ACTIONS.iter().copied().enumerate() {
            let row = index / 3;
            let column = index % 3;
            let rect = Rect::from_min_size(
                Pos2::new(column as f32 * 100.0 + 4.0, row as f32 * 100.0 + 4.0),
                Vec2::new(92.0, 92.0),
            );
            let response = ui.allocate_rect(rect, Sense::click());
            let selected = self.overlay_target == Some(OverlayTarget::Action(action));
            let fill = if selected {
                Color32::from_rgb(50, 112, 120)
            } else {
                Color32::from_rgba_unmultiplied(24, 40, 55, 235)
            };
            ui.painter().rect_filled(rect, 18.0, fill);
            ui.painter().rect_stroke(
                rect,
                18.0,
                Stroke::new(
                    2.0_f32,
                    if selected {
                        Color32::WHITE
                    } else {
                        Color32::from_gray(130)
                    },
                ),
                egui::StrokeKind::Outside,
            );
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                action_label(action),
                FontId::proportional(15.0),
                Color32::WHITE,
            );
            if response.clicked() {
                self.activate_action(action, context);
            }
        }
    }

    fn render_keyboard_overlay(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        for (row_index, row) in KEY_ROWS.iter().enumerate() {
            for (column, key) in row.iter().enumerate() {
                let rect = Rect::from_min_size(
                    Pos2::new(column as f32 * 72.0 + 3.0, row_index as f32 * 60.0 + 3.0),
                    Vec2::new(66.0, 54.0),
                );
                let action = KeyboardAction::Text(key);
                self.keyboard_button(ui, rect, key, action, context);
            }
        }
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(3.0, 183.0), Vec2::new(210.0, 72.0)),
            "SPACE",
            KeyboardAction::Text(" "),
            context,
        );
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(219.0, 183.0), Vec2::new(138.0, 72.0)),
            "BACK",
            KeyboardAction::Backspace,
            context,
        );
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(363.0, 183.0), Vec2::new(138.0, 72.0)),
            "ENTER",
            KeyboardAction::Enter,
            context,
        );
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(507.0, 183.0), Vec2::new(210.0, 72.0)),
            "CLOSE",
            KeyboardAction::Back,
            context,
        );
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(3.0, 263.0), Vec2::new(234.0, 72.0)),
            "the",
            KeyboardAction::Text("the "),
            context,
        );
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(243.0, 263.0), Vec2::new(234.0, 72.0)),
            "and",
            KeyboardAction::Text("and "),
            context,
        );
        let phrase = self
            .config
            .phrase_cards
            .first()
            .cloned()
            .unwrap_or_else(|| "thank you".to_owned());
        let phrase_action = if self.config.phrase_cards.is_empty() {
            KeyboardAction::Text("thank you")
        } else {
            KeyboardAction::Phrase(0)
        };
        self.keyboard_button(
            ui,
            Rect::from_min_size(Pos2::new(483.0, 263.0), Vec2::new(234.0, 72.0)),
            &phrase,
            phrase_action,
            context,
        );
    }

    fn keyboard_button(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        label: &str,
        action: KeyboardAction,
        context: &egui::Context,
    ) {
        let response = ui.allocate_rect(rect, Sense::click());
        let selected = self.overlay_target == Some(OverlayTarget::Key(action.clone()));
        ui.painter().rect_filled(
            rect,
            12.0,
            if selected {
                Color32::from_rgb(50, 112, 120)
            } else {
                Color32::from_rgba_unmultiplied(24, 40, 55, 235)
            },
        );
        ui.painter().rect_stroke(
            rect,
            12.0,
            Stroke::new(
                2.0_f32,
                if selected {
                    Color32::WHITE
                } else {
                    Color32::from_gray(135)
                },
            ),
            egui::StrokeKind::Outside,
        );
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(18.0),
            Color32::WHITE,
        );
        if response.clicked() {
            self.activate_keyboard_action(action, context);
        }
    }

    fn render_full_header(&mut self, ui: &mut egui::Ui, context: &egui::Context, title: &str) {
        ui.horizontal(|ui| {
            if ui.button("← Blob").clicked() {
                self.set_page(Page::Overlay, context);
            }
            ui.heading(title);
        });
        ui.separator();
    }

    fn render_training(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        self.render_full_header(ui, context, "Safe training environment");
        ui.label(
            "Training is always dry-run: no action below is sent to other Windows applications.",
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button("Practice click").clicked() {
                self.dispatch(InputAction::LeftClick);
            }
            if ui.button("Practice right click").clicked() {
                self.dispatch(InputAction::RightClick);
            }
            if ui.button("Practice double click").clicked() {
                self.dispatch(InputAction::DoubleClick);
            }
        });
        ui.add_space(8.0);
        ui.group(|ui| {
            ui.label(RichText::new("Drag practice").strong());
            if ui.button("Pick up").clicked() {
                self.dispatch(InputAction::LeftDown);
            }
            if ui.button("Drop safely").clicked() {
                self.dispatch(InputAction::LeftUp);
            }
            if ui.button("Simulate tracking loss").clicked() {
                self.dispatch(InputAction::LeftUp);
                let events = self.engine.set_paused(true);
                self.process_events(events);
            }
        });
        ui.group(|ui| {
            ui.label(RichText::new("Text practice").strong());
            ui.add(
                egui::TextEdit::multiline(&mut self.training_text)
                    .hint_text("Practice typing here…")
                    .desired_rows(3),
            );
            if ui.button("Record text injection").clicked() && !self.training_text.is_empty() {
                self.dispatch(InputAction::Text(self.training_text.clone()));
            }
        });
        self.render_activity(ui);
    }

    fn render_setup(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        ui.heading("EyeOS tracking workspace");
        ui.horizontal(|ui| {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(48.0), Sense::click());
            let colour = match self.engine.safety {
                SafetyState::Tracking => Color32::LIGHT_GREEN,
                SafetyState::TrackingLost => Color32::LIGHT_RED,
                SafetyState::Paused => Color32::YELLOW,
            };
            ui.painter()
                .circle_filled(rect.center(), 20.0, Color32::from_rgb(19, 31, 44));
            ui.painter()
                .circle_stroke(rect.center(), 20.0, Stroke::new(3.0_f32, colour));
            ui.painter().circle_filled(rect.center(), 5.0, colour);
            if response.clicked() {
                self.toggle_tracking();
            }
            ui.label(if self.has_validated_calibration() {
                "Precision validated"
            } else {
                "Calibration / validation needed"
            });
        });
        ui.group(|ui| {
            ui.label(RichText::new("Live tracker status").strong());
            ui.label(&self.status_message);
        });
        if let Some(progress) = self
            .tracker
            .as_ref()
            .and_then(|t| t.calibration_progress())
            .filter(|p| p.target.is_some())
        {
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "{}: target {} of {}",
                    progress.phase,
                    progress.completed + 1,
                    progress.total
                ))
                .strong(),
            );
            ui.label(&progress.instruction);
            ui.add(
                egui::ProgressBar::new(
                    progress.stable_samples as f32 / CALIBRATION_SAMPLES_PER_TARGET as f32,
                )
                .text(format!(
                    "{}/{} accepted samples",
                    progress.stable_samples, CALIBRATION_SAMPLES_PER_TARGET
                )),
            );
            ui.label(&progress.collection_status);
            ui.label(
                "Look at the highlighted circle. It moves after a stable fixation is collected.",
            );
            if ui.button("Cancel and return to setup").clicked() {
                if let Some(tracker) = self.tracker.as_mut() {
                    tracker.cancel_calibration();
                }
                self.set_page(Page::Setup, context);
            }
            return;
        }
        ui.add_space(6.0);
        ui.label("Place the camera at eye height with even lighting. Calibration must be completed by the intended user.");
        match &self.camera {
            CameraStatus::Available { devices } => ui.colored_label(
                Color32::LIGHT_GREEN,
                format!("Camera available ({devices} device(s) found)."),
            ),
            CameraStatus::Unavailable(message) => ui.colored_label(Color32::LIGHT_RED, message),
            CameraStatus::NotStarted => ui.label("Camera has not been checked."),
            CameraStatus::ModelMissing => ui.label("Camera is available but a model is missing."),
        };
        match self.model {
            ModelStatus::NotBundled => {
                ui.colored_label(
                    Color32::YELLOW,
                    "No reviewed local face/iris model is bundled.",
                );
                ui.label("EyeOS remains paused rather than guessing from webcam frames.");
            }
            ModelStatus::Ready => {
                ui.colored_label(Color32::LIGHT_GREEN, "Reviewed local model ready.");
            }
        }
        ui.separator();
        if let Some(profile) = &self.calibration {
            if let Some(report) = &profile.accuracy {
                ui.label(format!("Independent error: median {:.1}px, p95 {:.1}px; coverage {:.0}%; jitter {:.1}px; latency p95 {:.0}ms.",
                    report.median_error_px, report.p95_error_px, report.valid_sample_coverage * 100.0,
                    report.jitter_px, report.p95_latency_ms));
                if let (Some(median), Some(p95)) = (report.median_error_deg, report.p95_error_deg) {
                    ui.label(format!(
                        "Estimated angular error: median {median:.2}, p95 {p95:.2} degrees ({}).",
                        if profile.validation_passed {
                            "precision passed"
                        } else {
                            "extra calibration needed"
                        }
                    ));
                } else {
                    ui.label("Enter display dimensions and viewing distance to validate angular precision.");
                }
                ui.label(format!(
                    "Calibration retrieval: {}",
                    if profile.retrieval_enabled {
                        "enabled by grouped validation"
                    } else {
                        "baseline selected"
                    }
                ));
            }
        } else {
            ui.label("A fresh calibration is required for the new tracking engine.");
        }
        ui.label(format!(
            "Display: {:.0} × {:.0} pixels",
            self.screen_size.x, self.screen_size.y
        ));
        if let Some(display) = &self.detected_display {
            ui.label(format!(
                "Display-reported size: {:.0} × {:.0} mm (editable)",
                display.size_mm.x, display.size_mm.y
            ));
        } else {
            ui.label(
                "Display did not provide physical size; enter measured width and height below.",
            );
        }
        ui.label("Enter eye-to-screen distance in mm. Angular error is estimated from these measurements.");
        let mut width = self.config.screen_width_mm.unwrap_or(0.0);
        let mut height = self.config.screen_height_mm.unwrap_or(0.0);
        let mut distance = self.config.viewing_distance_mm.unwrap_or(0.0);
        let mut changed = ui
            .vertical(|ui| {
                let a = ui
                    .add(
                        egui::DragValue::new(&mut width)
                            .speed(1.0)
                            .prefix("Width mm: "),
                    )
                    .changed();
                let b = ui
                    .add(
                        egui::DragValue::new(&mut height)
                            .speed(1.0)
                            .prefix("Height mm: "),
                    )
                    .changed();
                let c = ui
                    .add(
                        egui::DragValue::new(&mut distance)
                            .speed(1.0)
                            .prefix("Distance mm: "),
                    )
                    .changed();
                a || b || c
            })
            .inner;
        if ui.button("Read display size automatically").clicked() {
            self.detected_display = primary_display_geometry();
            if let Some(display) = &self.detected_display {
                width = display.size_mm.x;
                height = display.size_mm.y;
                changed = true;
            } else {
                self.status_message = "This display does not expose usable physical dimensions; enter measured values.".into();
            }
        }
        if changed {
            self.config.screen_width_mm = (width > 0.0).then_some(width);
            self.config.screen_height_mm = (height > 0.0).then_some(height);
            self.config.viewing_distance_mm = (distance > 0.0).then_some(distance);
            let tracker_config = tracking_config(self.screen_size, &self.config);
            if let Some(tracker) = self.tracker.as_mut() {
                let _ = tracker.reconfigure(tracker_config);
            }
            self.calibration = None;
            let events = self.engine.set_paused(true);
            self.process_events(events);
            self.save_config();
        }
        let ready = self.model == ModelStatus::Ready
            && self.tracker.as_ref().is_some_and(|t| t.has_camera())
            && self.has_recent_eye_features();
        if !ready {
            ui.colored_label(
                Color32::YELLOW,
                "Waiting for a usable binocular eye stream.",
            );
        }
        if ui
            .add_enabled(
                ready,
                egui::Button::new("Start quick personalized calibration"),
            )
            .clicked()
        {
            match self.tracker.as_mut().unwrap().start_calibration() {
                Ok(()) => {
                    self.calibration = None;
                    let events = self.engine.set_paused(true);
                    self.process_events(events);
                    self.clear_overlay_target();
                    self.set_page(Page::Calibration, context);
                }
                Err(error) => self.status_message = error,
            }
        }
        let extra = self
            .tracker
            .as_ref()
            .and_then(|t| t.calibration_progress())
            .is_some_and(|p| p.target.is_none() && !p.suggested_targets.is_empty())
            || self.calibration.is_some();
        if ui
            .add_enabled(
                ready && extra,
                egui::Button::new(if self.calibration.is_some() {
                    "Add targeted calibration and revalidate"
                } else {
                    "Resume remaining calibration"
                }),
            )
            .clicked()
        {
            match self.tracker.as_mut().unwrap().extend_calibration() {
                Ok(()) => {
                    self.calibration = None;
                    let events = self.engine.set_paused(true);
                    self.process_events(events);
                    self.clear_overlay_target();
                    self.set_page(Page::Calibration, context);
                }
                Err(error) => self.status_message = error,
            }
        }
        if !extra {
            ui.label("Targeted calibration becomes available after initial calibration or an interrupted fixation.");
        }
        if self.has_validated_calibration() {
            ui.separator();
            if ui
                .checkbox(
                    &mut self.config.live_input_confirmed,
                    "Enable live desktop mouse control",
                )
                .changed()
            {
                if !self.config.live_input_confirmed {
                    self.input.set_dry_run(true);
                    let events = self.engine.set_paused(true);
                    self.process_events(events);
                }
                self.save_config();
            }
            if ui
                .button(if self.engine.safety == SafetyState::Tracking {
                    "Pause mouse control"
                } else {
                    "Start mouse control here"
                })
                .clicked()
            {
                if self.config.live_input_confirmed && self.engine.safety != SafetyState::Tracking {
                    self.input.set_dry_run(false);
                    let events = self.engine.set_paused(false);
                    self.process_events(events);
                } else {
                    let events = self.engine.set_paused(true);
                    self.process_events(events);
                    self.input.set_dry_run(true);
                }
            }
        }
        if ui.button("Re-check camera").clicked() {
            self.camera = detect_camera_status();
        }
        if self.has_validated_calibration() && ui.button("Open EyeOS desktop blob").clicked() {
            self.set_page(Page::Overlay, context);
        }
        ui.separator();
        ui.collapsing("Accessibility and safety settings", |ui| {
            self.render_settings(ui, context)
        });
    }

    fn render_workspace(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        let rect = ui.max_rect();
        ui.painter()
            .rect_filled(rect, 0.0, Color32::from_rgb(8, 16, 25));
        let target = self
            .tracker
            .as_ref()
            .and_then(|t| t.calibration_progress())
            .and_then(|p| p.target);
        if let Some(target) = target {
            let p = physical_to_logical(target, context.pixels_per_point());
            let p = Pos2::new(p.x as f32, p.y as f32);
            ui.painter()
                .circle_filled(p, 20.0, Color32::from_rgb(68, 230, 188));
            ui.painter()
                .circle_stroke(p, 32.0, Stroke::new(3.0_f32, Color32::WHITE));
        }
        if let Some(gaze) = self.gaze_preview {
            let gaze = physical_to_logical(gaze, context.pixels_per_point());
            ui.painter().circle_stroke(
                Pos2::new(gaze.x as f32, gaze.y as f32),
                10.0,
                Stroke::new(2.0_f32, Color32::from_rgb(255, 190, 80)),
            );
        }
        let width = 400.0_f32.min(rect.width() * 0.45);
        // Controls stay opposite the target; center and edge targets remain visible.
        let x = if target.is_some_and(|p| p.x < self.screen_size.x * 0.5) {
            rect.right() - width - 12.0
        } else {
            rect.left() + 12.0
        };
        egui::Area::new(egui::Id::new("tracking-workspace-controls"))
            .fixed_pos(Pos2::new(x, rect.top() + 12.0))
            .default_size(Vec2::new(width, rect.height() - 24.0))
            .show(context, |ui| {
                ui.set_min_height(rect.height() - 24.0);
                egui::Frame::new()
                    .fill(Color32::from_rgb(22, 34, 46))
                    .inner_margin(14.0)
                    .corner_radius(12.0)
                    .show(ui, |ui| {
                        ui.set_width(width - 28.0);
                        egui::ScrollArea::vertical()
                            .max_height(rect.height() - 52.0)
                            .show(ui, |ui| {
                                self.render_setup(ui, context);
                                ui.separator();
                                ui.label(if self.gaze_preview.is_some() {
                                    "Amber ring: measured gaze preview"
                                } else {
                                    "Gaze preview appears after the screen mapping is calibrated."
                                });
                                if ui.button("Close EyeOS").clicked() {
                                    context.send_viewport_cmd(egui::ViewportCommand::Close);
                                }
                            });
                    });
            });
    }

    fn render_settings(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        if self.page != Page::Setup {
            self.render_full_header(ui, context, "Accessibility and safety settings");
        }
        ui.add(egui::Slider::new(&mut self.config.overlay_opacity, 0.2..=1.0).text("Blob opacity"));
        ui.add(egui::Slider::new(&mut self.engine.dwell_ms, 250..=3_000).text("Click dwell (ms)"));
        ui.add(
            egui::Slider::new(&mut self.engine.keyboard_dwell_ms, 250..=3_000)
                .text("Keyboard dwell (ms)"),
        );
        ui.checkbox(&mut self.config.high_contrast, "High contrast");
        ui.checkbox(&mut self.config.sound_feedback, "Audio feedback");
        ui.separator();
        ui.label(RichText::new("Live desktop input").strong());
        let model_ready = self.model == ModelStatus::Ready && self.has_validated_calibration();
        ui.add_enabled_ui(model_ready, |ui| {
            ui.checkbox(
                &mut self.config.live_input_confirmed,
                "Caregiver confirms training is complete",
            );
            if ui.button("Enable live input").clicked() && self.config.live_input_confirmed {
                self.input.set_dry_run(false);
                let events = self.engine.set_paused(false);
                self.process_events(events);
            }
        });
        if self.input.is_dry_run() {
            ui.colored_label(
                Color32::LIGHT_GREEN,
                "Dry-run is active — no other application receives input.",
            );
        } else if ui.button("Return to dry-run now").clicked() {
            self.input.set_dry_run(true);
            let events = self.engine.set_paused(true);
            self.process_events(events);
        }
        if ui.button("Save settings").clicked() {
            self.save_config();
        }
    }

    fn render_activity(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(RichText::new("Recent safe actions").strong());
        for action in self.input.recent_events().rev().take(6) {
            ui.monospace(format!("{action:?}"));
        }
    }

    fn maybe_run_cursor_simulator(&mut self, context: &egui::Context) {
        if !self.simulate_gaze {
            return;
        }
        let Some(position) = physical_cursor_position() else {
            return;
        };
        let sample = GazeSample {
            position,
            confidence: 1.0,
            timestamp_ms: self.started_at.elapsed().as_millis() as u64,
        };
        self.process_gaze_sample(sample, context);
    }

    fn has_recent_eye_features(&self) -> bool {
        let now_ms = self.started_at.elapsed().as_millis() as u64;
        self.latest_features
            .is_some_and(|(_, timestamp_ms)| now_ms.saturating_sub(timestamp_ms) <= 2_000)
    }

    fn poll_tracker(&mut self, context: &egui::Context) {
        let events = match self.tracker.as_mut().map(|tracker| tracker.poll()) {
            Some(Ok(events)) => events,
            Some(Err(error)) => {
                self.status_message = error;
                let events = self.engine.update(GazeSample {
                    position: Point::default(),
                    confidence: 0.0,
                    timestamp_ms: self.started_at.elapsed().as_millis() as u64,
                });
                self.process_events(events);
                self.clear_overlay_target();
                return;
            }
            None => return,
        };
        for event in events {
            if let Some(observation) = event.observation {
                self.latest_features = Some((observation, observation.timestamp_ms));
            }
            if let Some(status) = event.status {
                self.status_message = tracker_status_message(status);
            }
            if let Some(outcome) = event.calibration {
                match outcome {
                    CalibrationOutcome::Completed { profile, report } => {
                        self.status_message = if report.precision_passed {
                            format!(
                                "Precision validated: median {:.1}px, p95 {:.1}px.",
                                report.median_error_px, report.p95_error_px
                            )
                        } else if report.median_error_deg.is_none() {
                            "Pixel accuracy measured. Enter display size/viewing distance and repeat calibration to enable precision control.".into()
                        } else {
                            "Precision target missed. Add targeted calibration and revalidate; desktop control remains paused.".into()
                        };
                        if let Err(error) = self.store.save_calibration(&profile) {
                            self.status_message = format!("Could not save calibration: {error}");
                            if let Some(tracker) = self.tracker.as_mut() {
                                tracker.engine_mut().clear_profile();
                            }
                        } else {
                            self.calibration = Some(profile);
                        }
                        self.set_page(Page::Setup, context);
                    }
                    CalibrationOutcome::Rejected(error) => {
                        self.status_message = error;
                        self.set_page(Page::Setup, context);
                    }
                }
            }
            if let Some(estimate) = event.estimate {
                self.gaze_preview = estimate.filtered;
                if estimate.state == TrackingState::Tracking && estimate.precision_validated {
                    if let Some(position) = estimate.filtered {
                        self.process_gaze_sample(
                            GazeSample {
                                position,
                                confidence: estimate.quality_score,
                                timestamp_ms: estimate.timestamp_ms,
                            },
                            context,
                        );
                    }
                } else {
                    self.clear_overlay_target();
                    if self.engine.safety != SafetyState::Paused {
                        let events = self.engine.update(GazeSample {
                            position: Point::default(),
                            confidence: 0.0,
                            timestamp_ms: estimate.timestamp_ms,
                        });
                        self.process_events(events);
                    }
                }
            }
        }
        if self.latest_features.is_some_and(|(_, t)| {
            (self.started_at.elapsed().as_millis() as u64).saturating_sub(t) > 200
        }) {
            self.gaze_preview = None;
            self.clear_overlay_target();
            if self.engine.safety != SafetyState::Paused {
                let events = self.engine.update(GazeSample {
                    position: Point::default(),
                    confidence: 0.0,
                    timestamp_ms: self.started_at.elapsed().as_millis() as u64,
                });
                self.process_events(events);
            }
        }
    }
}

impl eframe::App for EyeOsApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        context.request_repaint_after(Duration::from_millis(16));
        self.poll_tracker(context);
        self.maybe_run_cursor_simulator(context);
        if self.config.high_contrast {
            context.set_visuals(egui::Visuals::dark());
        }

        // An intentional mouse click is retained for caregivers who configure EyeOS with a
        // mouse. Day-to-day operation uses `process_gaze_sample` and needs no motor input.
        match self.page {
            Page::Overlay => {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        self.render_overlay(ui, context);
                    });
            }
            Page::Actions => {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        self.render_actions(ui, context);
                    });
            }
            Page::Keyboard => {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        self.render_keyboard_overlay(ui, context);
                    });
            }
            Page::Calibration | Page::Setup => {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| self.render_workspace(ui, context));
            }
            Page::Training => {
                egui::CentralPanel::default().show(context, |ui| self.render_training(ui, context));
            }
        }

        if context.input(|input| input.key_pressed(egui::Key::Escape)) {
            match self.page {
                Page::Overlay => self.toggle_tracking(),
                Page::Calibration => {
                    if let Some(tracker) = self.tracker.as_mut() {
                        tracker.cancel_calibration();
                    }
                    self.set_page(Page::Setup, context);
                }
                _ => self.set_page(Page::Overlay, context),
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let events = self.engine.set_paused(true);
        self.process_events(events);
        self.save_config();
    }
}

const ACTIONS: [OverlayAction; 9] = [
    OverlayAction::LeftClick,
    OverlayAction::DoubleClick,
    OverlayAction::RightClick,
    OverlayAction::Drag,
    OverlayAction::Scroll,
    OverlayAction::Keyboard,
    OverlayAction::Training,
    OverlayAction::Calibrate,
    OverlayAction::Pause,
];

const KEY_ROWS: [&[&str]; 3] = [
    &["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    &["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    &["z", "x", "c", "v", "b", "n", "m"],
];

fn action_label(action: OverlayAction) -> &'static str {
    match action {
        OverlayAction::LeftClick => "CLICK",
        OverlayAction::DoubleClick => "DOUBLE",
        OverlayAction::RightClick => "RIGHT",
        OverlayAction::Drag => "DRAG",
        OverlayAction::Scroll => "SCROLL",
        OverlayAction::Keyboard => "TYPE",
        OverlayAction::Training => "PRACTICE",
        OverlayAction::Calibrate => "SETUP",
        OverlayAction::Pause => "PAUSE\nRESUME",
    }
}

fn tracker_status_message(status: TrackerStatus) -> String {
    match status {
        TrackerStatus::Starting => {
            "Starting the local face, head-pose, and gaze-vector tracker…".to_owned()
        }
        TrackerStatus::CameraReady {
            width,
            height,
            fps,
            format,
            ..
        } => format!(
            "Camera streaming at {width}×{height}, {fps} FPS ({format}); looking for a face."
        ),
        TrackerStatus::CameraRetrying { attempt, detail } => {
            format!("Camera is reconnecting (attempt {attempt}): {detail}")
        }
        TrackerStatus::Tracking { fps } => format!("Eye tracker active ({fps:.0} FPS)."),
        TrackerStatus::LowFrameRate { fps } => {
            format!("Webcam processing at {fps:.0} FPS; calibration may take longer.")
        }
        TrackerStatus::GazeUnavailable { detail } => {
            format!("Tracking paused: gaze estimate is not reliable ({detail}).")
        }
        TrackerStatus::NoFace => "Tracking paused: face or eyes are not visible.".to_owned(),
        TrackerStatus::Failed(error) => format!("Eye tracker stopped: {error}"),
        TrackerStatus::Stopped => "Eye tracker stopped.".to_owned(),
    }
}

fn mode_label(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Pointer => "Left-click mode",
        InteractionMode::DoubleClick => "Double-click mode",
        InteractionMode::RightClick => "Right-click mode",
        InteractionMode::DragReady => "Drag mode: dwell on the source to pick up",
        InteractionMode::Dragging => "Drag in progress",
        InteractionMode::Scroll => "Scroll mode",
        InteractionMode::Keyboard => "Keyboard mode",
    }
}

fn primary_screen_size() -> Point {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN,
        };
        let width = unsafe { GetSystemMetrics(SM_CXSCREEN) }.max(1);
        let height = unsafe { GetSystemMetrics(SM_CYSCREEN) }.max(1);
        return Point::new(f64::from(width), f64::from(height));
    }
    #[cfg(not(windows))]
    Point::new(1920.0, 1080.0)
}

/// Physical display geometry is supplied explicitly; missing measurements never
/// become an inferred angular-precision claim.
fn tracking_config(screen_size: Point, config: &AppConfig) -> TrackerConfig {
    TrackerConfig {
        screen_size,
        screen_width_mm: config.screen_width_mm,
        screen_height_mm: config.screen_height_mm,
        viewing_distance_mm: config.viewing_distance_mm,
        camera_id: format!("webcam:{}", config.camera_index),
        display_id: format!("primary:{}x{}", screen_size.x, screen_size.y),
        ..TrackerConfig::default()
    }
}

// Window sizes are logical units. Query primary-display scaling before the
// first egui input frame; context.pixels_per_point() is not initialized yet.
fn primary_ui_size() -> Point {
    #[cfg(windows)]
    let scale = unsafe { windows_sys::Win32::UI::HiDpi::GetDpiForSystem() }.max(96) as f32 / 96.0;
    #[cfg(not(windows))]
    let scale = 1.0;
    physical_to_logical(primary_screen_size(), scale)
}

fn physical_cursor_position() -> Option<Point> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::{Foundation::POINT, UI::WindowsAndMessaging::GetCursorPos};
        let mut point = POINT { x: 0, y: 0 };
        if unsafe { GetCursorPos(&mut point) } != 0 {
            return Some(Point::new(f64::from(point.x), f64::from(point.y)));
        }
    }
    None
}

fn run() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    let cli = Cli::parse();
    let store = ProfileStore::for_current_user()?;
    if cli.reset_profile {
        store.reset().context("resetting the EyeOS profile")?;
        println!(
            "EyeOS settings and encrypted calibration were removed from {}",
            store.root().display()
        );
        return Ok(());
    }
    if cli.install_autostart {
        install_autostart()?;
        println!("EyeOS will start after this Windows user signs in.");
        return Ok(());
    }

    let config = store.load_config()?;
    let page = if cli.training {
        Page::Training
    } else {
        Page::Setup
    };
    let (size, position) = match page {
        Page::Overlay => (Vec2::splat(BLOB_SIZE), [OVERLAY_MARGIN, 900.0]),
        Page::Training => (Vec2::new(620.0, 620.0), [80.0, 80.0]),
        Page::Setup => {
            let size = primary_ui_size();
            (Vec2::new(size.x as f32, size.y as f32), [0.0, 0.0])
        }
        Page::Actions => (Vec2::splat(PANEL_SIZE), [OVERLAY_MARGIN, 700.0]),
        Page::Keyboard => (
            Vec2::new(KEYBOARD_WIDTH, KEYBOARD_HEIGHT),
            [OVERLAY_MARGIN, 600.0],
        ),
        Page::Calibration => (Vec2::new(1920.0, 1080.0), [0.0, 0.0]),
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("EyeOS")
            .with_inner_size(size)
            .with_position(position)
            .with_transparent(matches!(
                page,
                Page::Overlay | Page::Actions | Page::Keyboard
            ))
            .with_fullscreen(false)
            .with_decorations(false)
            .with_resizable(false)
            .with_always_on_top(),
        persist_window: false,
        ..Default::default()
    };
    eframe::run_native(
        "EyeOS",
        options,
        Box::new(move |context| {
            let mut app = EyeOsApp::new(store, config, page, cli.simulate_gaze);
            app.set_page(app.page, &context.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow::Error::msg(error.to_string()))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("EyeOS could not start: {error:#}");
        std::process::exit(1);
    }
}
