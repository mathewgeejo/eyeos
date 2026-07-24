use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eyeos::{
    CalibrationPoint, CalibrationProfile, ControlEngine, EngineEvent, GazeSample, InputAction,
    InputController, InteractionMode, Point, SafetyState,
    config::AppConfig,
    persistence::{ProfileStore, install_autostart},
    tracker::{LocalTracker, TrackerEvent, TrackerStatus},
    vision::{
        CameraStatus, EyeFeatures, ModelStatus, calibrated_sample, detect_camera_status,
        model_status,
    },
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
    Settings,
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

/// Each target settles for 900 ms, then collects 16 confident local samples. This keeps the
/// caregiver-assisted process keyboard-free for the person being calibrated.
struct CalibrationWizard {
    targets: Vec<Point>,
    samples: Vec<CalibrationPoint>,
    current: usize,
    settle_until_ms: Option<u64>,
    feature_samples: Vec<EyeFeatures>,
}

impl CalibrationWizard {
    fn new(screen_size: Point) -> Self {
        let mut targets = Vec::with_capacity(9);
        for y in [0.1, 0.5, 0.9] {
            for x in [0.1, 0.5, 0.9] {
                targets.push(Point::new(screen_size.x * x, screen_size.y * y));
            }
        }
        Self {
            targets,
            samples: Vec::with_capacity(9),
            current: 0,
            settle_until_ms: None,
            feature_samples: Vec::with_capacity(16),
        }
    }

    fn target(&self) -> Option<Point> {
        self.targets.get(self.current).copied()
    }

    fn progress(&self) -> (usize, usize) {
        (self.current, self.targets.len())
    }

    fn sample_progress(&self) -> usize {
        self.feature_samples.len()
    }

    fn observe(&mut self, features: EyeFeatures, timestamp_ms: u64) -> Option<CalibrationProfile> {
        if features.confidence < 0.72 || self.current >= self.targets.len() {
            return None;
        }
        let settle_until = self
            .settle_until_ms
            .get_or_insert(timestamp_ms.saturating_add(900));
        if timestamp_ms < *settle_until {
            return None;
        }
        self.feature_samples.push(features);
        if self.feature_samples.len() < 16 {
            return None;
        }

        let count = self.feature_samples.len() as f64;
        let feature_x = self
            .feature_samples
            .iter()
            .map(|sample| sample.x)
            .sum::<f64>()
            / count;
        let feature_y = self
            .feature_samples
            .iter()
            .map(|sample| sample.y)
            .sum::<f64>()
            / count;
        let target = self.targets[self.current];
        self.samples.push(CalibrationPoint {
            feature_x,
            feature_y,
            screen_x: target.x,
            screen_y: target.y,
        });
        self.current += 1;
        self.feature_samples.clear();
        self.settle_until_ms = Some(timestamp_ms.saturating_add(900));
        (self.current == self.targets.len())
            .then(|| CalibrationProfile::fit(&self.samples))
            .flatten()
    }
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
    tracker: Option<LocalTracker>,
    latest_features: Option<(EyeFeatures, u64)>,
    calibration_wizard: Option<CalibrationWizard>,
    overlay_target: Option<OverlayTarget>,
    overlay_target_started_at: Option<u64>,
    overlay_cooldown_until: u64,
}

impl EyeOsApp {
    fn new(store: ProfileStore, config: AppConfig, page: Page, simulate_gaze: bool) -> Self {
        let screen_size = primary_screen_size();
        let mut engine = ControlEngine::new(screen_size.x, screen_size.y);
        engine.dwell_ms = config.dwell_ms;
        engine.keyboard_dwell_ms = config.keyboard_dwell_ms;

        let calibration = store.load_calibration().ok().flatten();
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
            started_at: Instant::now(),
            simulate_gaze,
            training_text: String::new(),
            status_message: "Looking for the local eye tracker…".to_owned(),
            dwell_progress: 0.0,
            calibration,
            tracker: None,
            latest_features: None,
            calibration_wizard: None,
            overlay_target: None,
            overlay_target_started_at: None,
            overlay_cooldown_until: 0,
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
            match LocalTracker::start(app.store.root().to_path_buf(), app.config.camera_index) {
                Ok(tracker) => app.tracker = Some(tracker),
                Err(error) => {
                    app.status_message = format!("Eye tracker could not start: {error}");
                    return app;
                }
            }
        }

        if !app.simulate_gaze
            && app.model == ModelStatus::Ready
            && app.calibration.is_some()
            && app.config.live_input_confirmed
        {
            app.input.set_dry_run(false);
            let events = app.engine.set_paused(false);
            app.process_events(events);
        } else if !app.simulate_gaze {
            app.status_message = if app.calibration.is_none() {
                "Tracker starting — complete the one-time 9-point calibration.".to_owned()
            } else {
                "Paused: caregiver confirmation is required before live input.".to_owned()
            };
        }
        app
    }

    fn dispatch(&mut self, action: InputAction) {
        if let Err(error) = self.input.dispatch(action) {
            self.status_message = format!("Input was not sent: {error}");
        }
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
        let (size, position) = match page {
            Page::Overlay => (
                Vec2::splat(BLOB_SIZE),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (self.screen_size.y as f32 - BLOB_SIZE - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Actions => (
                Vec2::splat(PANEL_SIZE),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (self.screen_size.y as f32 - PANEL_SIZE - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Keyboard => (
                Vec2::new(KEYBOARD_WIDTH, KEYBOARD_HEIGHT),
                Pos2::new(
                    OVERLAY_MARGIN,
                    (self.screen_size.y as f32 - KEYBOARD_HEIGHT - OVERLAY_MARGIN).max(0.0),
                ),
            ),
            Page::Calibration => (
                Vec2::new(self.screen_size.x as f32, self.screen_size.y as f32),
                Pos2::ZERO,
            ),
            Page::Training | Page::Setup | Page::Settings => {
                (Vec2::new(620.0, 620.0), Pos2::new(80.0, 80.0))
            }
        };
        context.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
        context.send_viewport_cmd(egui::ViewportCommand::OuterPosition(position));
    }

    fn toggle_tracking(&mut self) {
        if self.model != ModelStatus::Ready || self.calibration.is_none() {
            self.status_message =
                "Tracking cannot start until a local model and calibration are available."
                    .to_owned();
            return;
        }
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
                && self.calibration.is_some()
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
            Page::Calibration | Page::Training | Page::Setup | Page::Settings => {}
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
        self.render_full_header(ui, context, "Caregiver setup and calibration");
        ui.group(|ui| {
            ui.label(RichText::new("Live tracker status").strong());
            ui.label(&self.status_message);
        });
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
            ui.label(format!(
                "Encrypted calibration profile: {} samples, median fit error {:.1}px.",
                profile.sample_count, profile.median_error_px
            ));
        } else {
            ui.label("No calibration profile is stored yet.");
        }
        let ready_for_calibration = self.model == ModelStatus::Ready
            && self.tracker.is_some()
            && self.has_recent_eye_features();
        if !ready_for_calibration {
            ui.colored_label(
                Color32::YELLOW,
                "Waiting for a usable camera and face/iris stream before calibration can begin.",
            );
        }
        ui.add_enabled_ui(ready_for_calibration, |ui| {
            if ui.button("Start 9-point calibration").clicked() {
                self.calibration_wizard = Some(CalibrationWizard::new(self.screen_size));
                self.status_message =
                    "Look at each target until EyeOS moves to the next one.".to_owned();
                self.set_page(Page::Calibration, context);
            }
        });
        if ui.button("Re-check camera").clicked() {
            self.camera = detect_camera_status();
        }
        ui.separator();
        if ui.button("Open accessibility settings").clicked() {
            self.set_page(Page::Settings, context);
        }
    }

    fn render_calibration(&mut self, ui: &mut egui::Ui) {
        let rect = ui.max_rect();
        ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
        let Some(wizard) = &self.calibration_wizard else {
            return;
        };
        let Some(target) = wizard.target() else {
            return;
        };
        let (completed, total) = wizard.progress();
        let target_position = Pos2::new(target.x as f32, target.y as f32);
        let target_colour = if self.config.high_contrast {
            Color32::WHITE
        } else {
            Color32::from_rgb(68, 230, 188)
        };
        ui.painter()
            .circle_filled(target_position, 28.0, target_colour);
        ui.painter()
            .circle_stroke(target_position, 42.0, Stroke::new(4.0_f32, Color32::WHITE));
        ui.painter().text(
            Pos2::new(rect.center().x, 48.0),
            Align2::CENTER_CENTER,
            format!("Look at the target  •  {} of {}", completed + 1, total),
            FontId::proportional(26.0),
            Color32::WHITE,
        );
        ui.painter().text(
            Pos2::new(rect.center().x, 82.0),
            Align2::CENTER_CENTER,
            format!("Capturing stable samples: {}/16", wizard.sample_progress()),
            FontId::proportional(18.0),
            Color32::from_gray(210),
        );
    }

    fn render_settings(&mut self, ui: &mut egui::Ui, context: &egui::Context) {
        self.render_full_header(ui, context, "Accessibility and safety settings");
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
        let model_ready = self.model == ModelStatus::Ready && self.calibration.is_some();
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
        let events = self
            .tracker
            .as_ref()
            .map(LocalTracker::drain)
            .unwrap_or_default();
        for event in events {
            match event {
                TrackerEvent::Features {
                    features,
                    timestamp_ms,
                } => {
                    self.latest_features = Some((features, timestamp_ms));
                    let completed_calibration = self
                        .calibration_wizard
                        .as_mut()
                        .and_then(|wizard| wizard.observe(features, timestamp_ms));
                    if let Some(calibration) = completed_calibration {
                        match self.store.save_calibration(&calibration) {
                            Ok(()) => {
                                self.calibration = Some(calibration);
                                self.calibration_wizard = None;
                                self.status_message =
                                    "Calibration saved. A caregiver can now enable live input."
                                        .to_owned();
                                self.set_page(Page::Setup, context);
                            }
                            Err(error) => {
                                self.calibration_wizard = None;
                                self.status_message =
                                    format!("Could not save calibration: {error}");
                                self.set_page(Page::Setup, context);
                            }
                        }
                    } else if self.calibration_wizard.is_none() {
                        if let Some(calibration) = &self.calibration {
                            self.process_gaze_sample(
                                calibrated_sample(features, calibration, timestamp_ms),
                                context,
                            );
                        }
                    }
                }
                TrackerEvent::Status(status) => {
                    let tracking_failed = matches!(
                        status,
                        TrackerStatus::NoFace
                            | TrackerStatus::LowFrameRate { .. }
                            | TrackerStatus::Failed(_)
                    );
                    if tracking_failed && self.engine.safety != SafetyState::Paused {
                        let timestamp_ms = self
                            .latest_features
                            .map(|(_, timestamp_ms)| timestamp_ms)
                            .unwrap_or_else(|| self.started_at.elapsed().as_millis() as u64);
                        let events = self.engine.update(GazeSample {
                            position: Point::default(),
                            confidence: 0.0,
                            timestamp_ms,
                        });
                        self.process_events(events);
                    }
                    self.status_message = tracker_status_message(status);
                }
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
            Page::Calibration => {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| self.render_calibration(ui));
            }
            Page::Training => {
                egui::CentralPanel::default().show(context, |ui| self.render_training(ui, context));
            }
            Page::Setup => {
                egui::CentralPanel::default().show(context, |ui| self.render_setup(ui, context));
            }
            Page::Settings => {
                egui::CentralPanel::default().show(context, |ui| self.render_settings(ui, context));
            }
        }

        if context.input(|input| input.key_pressed(egui::Key::Escape)) {
            match self.page {
                Page::Overlay => self.toggle_tracking(),
                Page::Calibration => {
                    self.calibration_wizard = None;
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
        TrackerStatus::Starting => "Starting the local MediaPipe eye tracker…".to_owned(),
        TrackerStatus::CameraReady {
            width,
            height,
            fps,
            format,
        } => format!(
            "Camera streaming at {width}×{height}, {fps} FPS ({format}); looking for a face."
        ),
        TrackerStatus::CameraRetrying { attempt, detail } => {
            format!("Camera is reconnecting (attempt {attempt}): {detail}")
        }
        TrackerStatus::Tracking { fps } => format!("Eye tracker active ({fps:.0} FPS)."),
        TrackerStatus::LowFrameRate { fps } => {
            format!("Tracking paused: webcam inference is too slow ({fps:.0} FPS).")
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
    } else if cli.setup {
        Page::Setup
    } else {
        Page::Overlay
    };
    let (size, position) = match page {
        Page::Overlay => (Vec2::splat(BLOB_SIZE), [OVERLAY_MARGIN, 900.0]),
        Page::Training | Page::Setup | Page::Settings => (Vec2::new(620.0, 620.0), [80.0, 80.0]),
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
            .with_transparent(true)
            .with_decorations(false)
            .with_resizable(false)
            .with_always_on_top(),
        ..Default::default()
    };
    eframe::run_native(
        "EyeOS",
        options,
        Box::new(move |context| {
            let mut app = EyeOsApp::new(store, config, page, cli.simulate_gaze);
            app.set_page(page, &context.egui_ctx);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_wizard_fits_from_observed_eye_features() {
        let mut wizard = CalibrationWizard::new(Point::new(1_920.0, 1_080.0));
        let mut completed = None;
        let mut timestamp_ms = 0_u64;
        while wizard.target().is_some() {
            let target = wizard.target().expect("calibration target");
            let features = EyeFeatures {
                x: target.x / 1_920.0,
                y: target.y / 1_080.0,
                confidence: 1.0,
                one_eye_fallback: false,
            };
            assert!(wizard.observe(features, timestamp_ms).is_none());
            timestamp_ms += 1_000;
            for _ in 0..16 {
                completed = wizard.observe(features, timestamp_ms);
                timestamp_ms += 33;
            }
        }
        let profile = completed.expect("completed profile");
        assert_eq!(profile.sample_count, 9);
        assert!(profile.median_error_px < 0.01);
    }
}
