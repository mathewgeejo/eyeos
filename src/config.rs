use serde::{Deserialize, Serialize};

pub const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub version: u32,
    pub overlay_opacity: f32,
    pub high_contrast: bool,
    pub sound_feedback: bool,
    pub dwell_ms: u64,
    pub keyboard_dwell_ms: u64,
    pub camera_index: u32,
    pub start_dry_run: bool,
    pub live_input_confirmed: bool,
    pub phrase_cards: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            overlay_opacity: 0.82,
            high_contrast: false,
            sound_feedback: false,
            dwell_ms: 650,
            keyboard_dwell_ms: 500,
            camera_index: 0,
            start_dry_run: true,
            live_input_confirmed: false,
            phrase_cards: vec![
                "Please give me a moment.".to_owned(),
                "Thank you.".to_owned(),
            ],
        }
    }
}

impl AppConfig {
    pub fn migrate(mut self) -> Self {
        // Future migrations deliberately preserve user-safe defaults.
        if self.version == 0 {
            self.version = CONFIG_VERSION;
            self.start_dry_run = true;
            self.live_input_confirmed = false;
        }
        self.overlay_opacity = self.overlay_opacity.clamp(0.2, 1.0);
        self.dwell_ms = self.dwell_ms.clamp(250, 3_000);
        self.keyboard_dwell_ms = self.keyboard_dwell_ms.clamp(250, 3_000);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_preserves_safe_defaults() {
        let migrated = AppConfig {
            version: 0,
            start_dry_run: false,
            live_input_confirmed: true,
            ..AppConfig::default()
        }
        .migrate();
        assert_eq!(migrated.version, CONFIG_VERSION);
        assert!(migrated.start_dry_run);
        assert!(!migrated.live_input_confirmed);
    }
}
