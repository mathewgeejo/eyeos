use serde::{Deserialize, Serialize};

use crate::input::InputAction;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    fn lerp(self, target: Self, alpha: f64) -> Self {
        Self::new(
            self.x + (target.x - self.x) * alpha,
            self.y + (target.y - self.y) * alpha,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GazeSample {
    pub position: Point,
    pub confidence: f32,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum InteractionMode {
    #[default]
    Pointer,
    DoubleClick,
    RightClick,
    DragReady,
    Dragging,
    Scroll,
    Keyboard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyState {
    Paused,
    Tracking,
    TrackingLost,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Action(InputAction),
    SafetyChanged(SafetyState),
    DwellProgress(f32),
}

/// A conservative state machine: confidence loss always wins over convenience. It only emits
/// high-risk actions after a fixation followed by a dwell.
#[derive(Debug, Clone)]
pub struct ControlEngine {
    pub mode: InteractionMode,
    pub safety: SafetyState,
    pub dwell_ms: u64,
    pub keyboard_dwell_ms: u64,
    pub fixation_ms: u64,
    pub cooldown_ms: u64,
    pub minimum_confidence: f32,
    pub screen_size: Point,
    smoothed: Option<Point>,
    fixation_point: Option<Point>,
    fixation_started_at: Option<u64>,
    cooldown_until: u64,
    drag_is_held: bool,
}

impl ControlEngine {
    pub fn new(screen_width: f64, screen_height: f64) -> Self {
        Self {
            mode: InteractionMode::Pointer,
            safety: SafetyState::Paused,
            dwell_ms: 650,
            keyboard_dwell_ms: 500,
            fixation_ms: 120,
            cooldown_ms: 250,
            minimum_confidence: 0.72,
            screen_size: Point::new(screen_width, screen_height),
            smoothed: None,
            fixation_point: None,
            fixation_started_at: None,
            cooldown_until: 0,
            drag_is_held: false,
        }
    }

    pub fn set_paused(&mut self, paused: bool) -> Vec<EngineEvent> {
        if paused && self.drag_is_held {
            self.drag_is_held = false;
            self.mode = InteractionMode::Pointer;
            self.clear_fixation();
            self.safety = SafetyState::Paused;
            return vec![
                EngineEvent::Action(InputAction::LeftUp),
                EngineEvent::SafetyChanged(SafetyState::Paused),
            ];
        }

        let desired = if paused {
            SafetyState::Paused
        } else {
            SafetyState::Tracking
        };
        self.clear_fixation();
        if self.safety != desired {
            self.safety = desired;
            vec![EngineEvent::SafetyChanged(desired)]
        } else {
            Vec::new()
        }
    }

    pub fn set_mode(&mut self, mode: InteractionMode) {
        if !self.drag_is_held && mode != InteractionMode::Dragging {
            self.mode = mode;
        }
        self.clear_fixation();
    }

    pub fn update(&mut self, sample: GazeSample) -> Vec<EngineEvent> {
        if sample.confidence < self.minimum_confidence {
            return self.lose_tracking();
        }

        if self.safety == SafetyState::TrackingLost {
            self.safety = SafetyState::Paused;
            self.clear_fixation();
            return vec![EngineEvent::SafetyChanged(SafetyState::Paused)];
        }
        if self.safety == SafetyState::Paused {
            return Vec::new();
        }

        let target = Point::new(
            sample.position.x.clamp(0.0, self.screen_size.x),
            sample.position.y.clamp(0.0, self.screen_size.y),
        );
        let smooth = self
            .smoothed
            .map(|previous| previous.lerp(target, 0.35))
            .unwrap_or(target);
        self.smoothed = Some(smooth);

        let mut events = Vec::new();
        match self.mode {
            InteractionMode::Scroll => {
                let center = self.screen_size.y / 2.0;
                let zone = self.screen_size.y * 0.18;
                let displacement = smooth.y - center;
                if displacement.abs() > zone {
                    let direction = displacement.signum();
                    let amount = ((displacement.abs() - zone) / zone * 3.0).clamp(1.0, 3.0);
                    events.push(EngineEvent::Action(InputAction::ScrollLines(
                        (direction * amount).round() as i32,
                    )));
                }
                self.clear_fixation();
                return events;
            }
            InteractionMode::Keyboard => return events,
            _ => events.push(EngineEvent::Action(InputAction::MoveTo(smooth))),
        }

        // A movement larger than 90 logical pixels is considered a new aim, not a dwell.
        let stable = self
            .fixation_point
            .is_some_and(|fixation| fixation.distance_to(smooth) <= 90.0);
        if !stable {
            self.fixation_point = Some(smooth);
            self.fixation_started_at = Some(sample.timestamp_ms);
            events.push(EngineEvent::DwellProgress(0.0));
            return events;
        }

        let started = self.fixation_started_at.unwrap_or(sample.timestamp_ms);
        let elapsed = sample.timestamp_ms.saturating_sub(started);
        if elapsed < self.fixation_ms {
            events.push(EngineEvent::DwellProgress(0.0));
            return events;
        }

        let dwell_elapsed = elapsed - self.fixation_ms;
        let dwell = self.dwell_ms;
        events.push(EngineEvent::DwellProgress(
            (dwell_elapsed as f32 / dwell as f32).clamp(0.0, 1.0),
        ));
        if sample.timestamp_ms < self.cooldown_until || dwell_elapsed < dwell {
            return events;
        }

        let action = match self.mode {
            InteractionMode::Pointer => Some(InputAction::LeftClick),
            InteractionMode::DoubleClick => {
                self.mode = InteractionMode::Pointer;
                Some(InputAction::DoubleClick)
            }
            InteractionMode::RightClick => {
                self.mode = InteractionMode::Pointer;
                Some(InputAction::RightClick)
            }
            InteractionMode::DragReady => {
                self.drag_is_held = true;
                self.mode = InteractionMode::Dragging;
                Some(InputAction::LeftDown)
            }
            InteractionMode::Dragging => {
                self.drag_is_held = false;
                self.mode = InteractionMode::Pointer;
                Some(InputAction::LeftUp)
            }
            InteractionMode::Scroll | InteractionMode::Keyboard => None,
        };
        if let Some(action) = action {
            events.push(EngineEvent::Action(action));
        }
        self.cooldown_until = sample.timestamp_ms + self.cooldown_ms;
        self.clear_fixation();
        events
    }

    fn lose_tracking(&mut self) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        if self.drag_is_held {
            self.drag_is_held = false;
            self.mode = InteractionMode::Pointer;
            events.push(EngineEvent::Action(InputAction::LeftUp));
        }
        self.clear_fixation();
        if self.safety != SafetyState::TrackingLost {
            self.safety = SafetyState::TrackingLost;
            events.push(EngineEvent::SafetyChanged(SafetyState::TrackingLost));
        }
        events
    }

    fn clear_fixation(&mut self) {
        self.fixation_point = None;
        self.fixation_started_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(x: f64, y: f64, confidence: f32, timestamp_ms: u64) -> GazeSample {
        GazeSample {
            position: Point::new(x, y),
            confidence,
            timestamp_ms,
        }
    }

    #[test]
    fn requires_a_fixation_and_dwell_before_clicking() {
        let mut engine = ControlEngine::new(1920.0, 1080.0);
        engine.set_paused(false);
        assert!(
            !engine
                .update(sample(500.0, 500.0, 1.0, 0))
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftClick)))
        );
        assert!(
            !engine
                .update(sample(500.0, 500.0, 1.0, 769))
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftClick)))
        );
        assert!(
            engine
                .update(sample(500.0, 500.0, 1.0, 770))
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftClick)))
        );
    }

    #[test]
    fn tracking_loss_releases_drag_and_pauses() {
        let mut engine = ControlEngine::new(1920.0, 1080.0);
        engine.set_paused(false);
        engine.set_mode(InteractionMode::DragReady);
        engine.update(sample(100.0, 100.0, 1.0, 0));
        let start = engine.update(sample(100.0, 100.0, 1.0, 770));
        assert!(
            start
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftDown)))
        );
        let stopped = engine.update(sample(100.0, 100.0, 0.0, 800));
        assert!(
            stopped
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftUp)))
        );
        assert_eq!(engine.safety, SafetyState::TrackingLost);
    }

    #[test]
    fn cooldown_prevents_duplicate_clicks() {
        let mut engine = ControlEngine::new(1920.0, 1080.0);
        engine.set_paused(false);
        engine.update(sample(100.0, 100.0, 1.0, 0));
        engine.update(sample(100.0, 100.0, 1.0, 770));
        engine.update(sample(100.0, 100.0, 1.0, 771));
        let follow_up = engine.update(sample(100.0, 100.0, 1.0, 800));
        assert!(
            !follow_up
                .iter()
                .any(|event| matches!(event, EngineEvent::Action(InputAction::LeftClick)))
        );
    }
}
