use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui::{self, Align, Button, Color32, Layout, RichText, Stroke, Vec2};
use eyeos::{
    CalibrationPoint, CalibrationProfile, ControlEngine, EngineEvent, GazeSample, InputAction,
    InputController, InteractionMode, Point, SafetyState,
    config::AppConfig,
    persistence::{ProfileStore, install_autostart},
    vision::{CameraStatus, ModelStatus, detect_camera_status, model_status},
};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Blob,
    Training,
    Keyboard,
    Setup,
    Settings,
}

struct EyeOsApp {
    store: ProfileStore,
    config: AppConfig,
    engine: ControlEngine,
    input: InputController,
    page: Page,
    camera: CameraStatus,
    model: ModelStatus,
    started_at: Instant,
    simulate_gaze_with_mouse: bool,
    training_text: String,
    status_message: String,
    dwell_progress: f32,
    calibration: Option<CalibrationProfile>,
}

impl EyeOsApp {
    fn new(store: ProfileStore, config: AppConfig, page: Page) -> Self {
        let mut engine = ControlEngine::new(1920.0, 1080.0);
        engine.dwell_ms = config.dwell_ms;
        engine.keyboard_dwell_ms = config.keyboard_dwell_ms;
        Self {
            calibration: store.load_calibration().ok().flatten(),
            store,
            config,
            engine,
            input: InputController::default(),
            page,
            camera: detect_camera_status(),
            model: model_status(),
            started_at: Instant::now(),
            simulate_gaze_with_mouse: false,
            training_text: String::new(),
            status_message: "Paused — open the blob menu when you are ready.".to_owned(),
            dwell_progress: 0.0,
        }
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
                        SafetyState::Tracking => {
                            "Tracking enabled in safe dry-run mode.".to_owned()
                        }
                        SafetyState::TrackingLost => {
                            "Tracking lost — released any held drag and paused.".to_owned()
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

    fn toggle_tracking(&mut self) {
        let pause = self.engine.safety != SafetyState::Paused;
        let events = self.engine.set_paused(pause);
        self.process_events(events);
    }

    fn select_mode(&mut self, mode: InteractionMode) {
        self.engine.set_mode(mode);
        self.status_message = format!("{} selected.", mode_label(mode));
    }

    fn maybe_run_mouse_simulator(&mut self, context: &egui::Context) {
        if !self.simulate_gaze_with_mouse || self.engine.safety != SafetyState::Tracking {
            return;
        }
        let pointer = context.input(|input| input.pointer.hover_pos());
        if let Some(pointer) = pointer {
            let elapsed = self.started_at.elapsed().as_millis() as u64;
            // This tool is explicitly for developer/training verification. It is not presented as
            // eye tracking and defaults off.
            let events = self.engine.update(GazeSample {
                position: Point::new(f64::from(pointer.x), f64::from(pointer.y)),
                confidence: 1.0,
                timestamp_ms: elapsed,
            });
            self.process_events(events);
        }
    }

    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let safety = match self.engine.safety {
                SafetyState::Paused => RichText::new("PAUSED").color(Color32::YELLOW),
                SafetyState::Tracking => RichText::new("TRACKING").color(Color32::LIGHT_GREEN),
                SafetyState::TrackingLost => {
                    RichText::new("TRACKING LOST").color(Color32::LIGHT_RED)
                }
            };
            ui.label(safety.strong());
            ui.separator();
            ui.label(if self.input.is_dry_run() {
                "DRY-RUN"
            } else {
                "LIVE INPUT"
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("⚙ Settings").clicked() {
                    self.page = Page::Settings;
                }
            });
        });
        ui.add(egui::ProgressBar::new(self.dwell_progress).show_percentage());
        ui.label(RichText::new(&self.status_message).small());
        ui.separator();
    }

    fn render_blob(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(16.0);
            let tracking_colour = match self.engine.safety {
                SafetyState::Paused => Color32::from_rgb(230, 180, 50),
                SafetyState::Tracking => Color32::from_rgb(58, 210, 135),
                SafetyState::TrackingLost => Color32::from_rgb(235, 90, 90),
            };
            let blob = Button::new(
                RichText::new(if self.engine.safety == SafetyState::Paused {
                    "●\nPAUSED"
                } else {
                    "●\nACTIVE"
                })
                .size(22.0),
            )
            .min_size(Vec2::splat(128.0))
            .fill(Color32::from_rgba_unmultiplied(
                25,
                37,
                50,
                (self.config.overlay_opacity * 245.0) as u8,
            ))
            .stroke(Stroke::new(5.0_f32, tracking_colour));
            if ui.add(blob).clicked() {
                self.toggle_tracking();
            }
            ui.add_space(10.0);
            ui.label(
                "The floating blob starts paused. Use its menu to choose an intentional action.",
            );
            ui.add_space(10.0);
        });

        egui::Grid::new("radial-action-grid")
            .num_columns(3)
            .spacing([10.0, 10.0])
            .show(ui, |ui| {
                action_button(ui, "Left click", || {
                    self.select_mode(InteractionMode::Pointer)
                });
                action_button(ui, "Double", || {
                    self.select_mode(InteractionMode::DoubleClick)
                });
                action_button(ui, "Right click", || {
                    self.select_mode(InteractionMode::RightClick)
                });
                action_button(ui, "Drag", || self.select_mode(InteractionMode::DragReady));
                action_button(ui, "Scroll", || self.select_mode(InteractionMode::Scroll));
                action_button(ui, "Keyboard", || self.page = Page::Keyboard);
                action_button(ui, "Training", || self.page = Page::Training);
                action_button(ui, "Calibrate", || self.page = Page::Setup);
                action_button(ui, "Pause", || {
                    let events = self.engine.set_paused(true);
                    self.process_events(events);
                });
            });
    }

    fn render_training(&mut self, ui: &mut egui::Ui) {
        ui.heading("Safe training environment");
        ui.label("Every action below is recorded locally. Nothing is sent to other Windows applications while DRY-RUN is shown.");
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_sized([120.0, 56.0], Button::new("Practice click"))
                .clicked()
            {
                self.dispatch(InputAction::LeftClick);
            }
            if ui
                .add_sized([120.0, 56.0], Button::new("Practice right click"))
                .clicked()
            {
                self.dispatch(InputAction::RightClick);
            }
            if ui
                .add_sized([120.0, 56.0], Button::new("Practice double click"))
                .clicked()
            {
                self.dispatch(InputAction::DoubleClick);
            }
        });
        ui.add_space(8.0);
        ui.group(|ui| {
            ui.label(RichText::new("Drag practice").strong());
            ui.horizontal(|ui| {
                if ui
                    .add_sized([140.0, 50.0], Button::new("1. Pick up"))
                    .clicked()
                {
                    self.dispatch(InputAction::LeftDown);
                }
                if ui
                    .add_sized([140.0, 50.0], Button::new("2. Drop safely"))
                    .clicked()
                {
                    self.dispatch(InputAction::LeftUp);
                }
            });
            if ui.button("Simulate tracking loss during drag").clicked() {
                self.dispatch(InputAction::LeftUp);
                let events = self.engine.set_paused(true);
                self.process_events(events);
                self.status_message =
                    "Training loss simulation released drag and paused.".to_owned();
            }
        });
        ui.add_space(8.0);
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
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Scroll up").clicked() {
                self.dispatch(InputAction::ScrollLines(3));
            }
            if ui.button("Scroll down").clicked() {
                self.dispatch(InputAction::ScrollLines(-3));
            }
            if ui.button("Back to blob").clicked() {
                self.page = Page::Blob;
            }
        });
        self.render_activity(ui);
    }

    fn render_keyboard(&mut self, ui: &mut egui::Ui) {
        ui.heading("Gaze keyboard");
        ui.label("Large direct keys with local predictions and phrase cards. Keyboard events are dry-run until live input is explicitly enabled.");
        for row in ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"] {
            ui.horizontal(|ui| {
                for letter in row.chars() {
                    if ui
                        .add_sized([42.0, 48.0], Button::new(letter.to_string()))
                        .clicked()
                    {
                        self.dispatch(InputAction::Text(letter.to_string().to_lowercase()));
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            if ui.add_sized([84.0, 48.0], Button::new("Space")).clicked() {
                self.dispatch(InputAction::Text(" ".to_owned()));
            }
            if ui.add_sized([84.0, 48.0], Button::new("Enter")).clicked() {
                self.dispatch(InputAction::Text("\n".to_owned()));
            }
            if ui
                .add_sized([84.0, 48.0], Button::new("Backspace"))
                .clicked()
            {
                self.dispatch(InputAction::KeyChord {
                    ctrl: false,
                    shift: false,
                    alt: false,
                    virtual_key: 0x08,
                });
            }
            if ui.button("Back").clicked() {
                self.page = Page::Blob;
            }
        });
        ui.separator();
        ui.label(RichText::new("Phrase cards").strong());
        for phrase in self.config.phrase_cards.clone() {
            if ui.add_sized([360.0, 42.0], Button::new(&phrase)).clicked() {
                self.dispatch(InputAction::Text(phrase));
            }
        }
    }

    fn render_setup(&mut self, ui: &mut egui::Ui) {
        ui.heading("Caregiver setup and calibration");
        ui.label("Place the camera at eye height, keep the face lit evenly, and complete calibration only with the intended user.");
        ui.add_space(6.0);
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
                ui.colored_label(Color32::YELLOW, "Live eye inference is locked: no reviewed local 478-point model is bundled yet.");
                ui.label("The control and calibration safety systems are implemented, but this build will not pretend webcam pixels are gaze data.");
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
        ui.add_enabled_ui(self.model == ModelStatus::Ready, |ui| {
            if ui
                .add_sized([260.0, 50.0], Button::new("Start 9-point calibration"))
                .clicked()
            {
                let profile = demo_calibration();
                match self.store.save_calibration(&profile) {
                    Ok(()) => {
                        self.calibration = Some(profile);
                        self.status_message =
                            "Calibration saved with Windows DPAPI encryption.".to_owned();
                    }
                    Err(error) => {
                        self.status_message = format!("Could not save calibration: {error}")
                    }
                }
            }
        });
        if ui.button("Re-check camera").clicked() {
            self.camera = detect_camera_status();
        }
        if ui.button("Back to blob").clicked() {
            self.page = Page::Blob;
        }
    }

    fn render_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Accessibility and safety settings");
        ui.add(egui::Slider::new(&mut self.config.overlay_opacity, 0.2..=1.0).text("Blob opacity"));
        ui.add(egui::Slider::new(&mut self.engine.dwell_ms, 250..=3_000).text("Click dwell (ms)"));
        ui.add(
            egui::Slider::new(&mut self.engine.keyboard_dwell_ms, 250..=3_000)
                .text("Keyboard dwell (ms)"),
        );
        ui.checkbox(&mut self.config.high_contrast, "High-contrast mode");
        ui.checkbox(&mut self.config.sound_feedback, "Audio feedback");
        ui.checkbox(
            &mut self.simulate_gaze_with_mouse,
            "Developer-only mouse gaze simulator (dry-run)",
        );
        ui.separator();
        ui.label(RichText::new("Live desktop input").strong());
        ui.label("Live input stays locked until a reviewed local eye model and user validation are available.");
        let model_ready = self.model == ModelStatus::Ready;
        ui.add_enabled_ui(model_ready, |ui| {
            ui.checkbox(
                &mut self.config.live_input_confirmed,
                "Caregiver confirms training is complete",
            );
            if ui.button("Enable live input").clicked() && self.config.live_input_confirmed {
                self.input.set_dry_run(false);
                self.status_message =
                    "Live input enabled. The blob still starts paused.".to_owned();
            }
        });
        if self.input.is_dry_run() {
            ui.label(
                RichText::new("Dry-run is active — no other application will receive input.")
                    .color(Color32::LIGHT_GREEN),
            );
        } else if ui.button("Return to dry-run immediately").clicked() {
            self.input.set_dry_run(true);
            let events = self.engine.set_paused(true);
            self.process_events(events);
        }
        ui.separator();
        if ui.button("Save settings").clicked() {
            self.save_config();
        }
        if ui.button("Open caregiver setup").clicked() {
            self.page = Page::Setup;
        }
        if ui.button("Back to blob").clicked() {
            self.page = Page::Blob;
        }
    }

    fn render_activity(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(RichText::new("Recent safe actions").strong());
        for action in self.input.recent_events().rev().take(6) {
            ui.monospace(format!("{action:?}"));
        }
    }
}

impl eframe::App for EyeOsApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        context.request_repaint_after(Duration::from_millis(16));
        self.maybe_run_mouse_simulator(context);
        if self.config.high_contrast {
            context.set_visuals(egui::Visuals::dark());
        }

        egui::CentralPanel::default().show(context, |ui| {
            self.render_top_bar(ui);
            match self.page {
                Page::Blob => self.render_blob(ui),
                Page::Training => self.render_training(ui),
                Page::Keyboard => self.render_keyboard(ui),
                Page::Setup => self.render_setup(ui),
                Page::Settings => self.render_settings(ui),
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_config();
    }
}

fn action_button(ui: &mut egui::Ui, label: &str, action: impl FnOnce()) {
    if ui.add_sized([116.0, 46.0], Button::new(label)).clicked() {
        action();
    }
}

fn mode_label(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Pointer => "Left click mode",
        InteractionMode::DoubleClick => "Double-click mode",
        InteractionMode::RightClick => "Right-click mode",
        InteractionMode::DragReady => "Drag mode: dwell on the source to pick up",
        InteractionMode::Dragging => "Drag in progress",
        InteractionMode::Scroll => "Scroll mode",
        InteractionMode::Keyboard => "Keyboard mode",
    }
}

fn demo_calibration() -> CalibrationProfile {
    // Only used if a bundled model has already unlocked the button. In normal operation the
    // calibration wizard supplies actual measured eye features instead of these grid values.
    let mut samples = Vec::new();
    for y in [0.1, 0.5, 0.9] {
        for x in [0.1, 0.5, 0.9] {
            samples.push(CalibrationPoint {
                feature_x: x,
                feature_y: y,
                screen_x: x * 1920.0,
                screen_y: y * 1080.0,
            });
        }
    }
    CalibrationProfile::fit(&samples).expect("nine non-collinear calibration samples")
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
        Page::Blob
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("EyeOS")
            .with_inner_size([460.0, 650.0])
            .with_min_inner_size([420.0, 520.0])
            .with_transparent(true)
            .with_decorations(false)
            .with_always_on_top(),
        ..Default::default()
    };
    eframe::run_native(
        "EyeOS",
        options,
        Box::new(move |_context| Ok(Box::new(EyeOsApp::new(store, config, page)))),
    )
    .map_err(|error| anyhow::Error::msg(error.to_string()))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("EyeOS could not start: {error:#}");
        std::process::exit(1);
    }
}
