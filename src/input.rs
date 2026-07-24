use std::collections::VecDeque;

use anyhow::Result;

use crate::gaze::Point;

#[derive(Debug, Clone, PartialEq)]
pub enum InputAction {
    MoveTo(Point),
    LeftClick,
    DoubleClick,
    RightClick,
    LeftDown,
    LeftUp,
    ScrollLines(i32),
    Text(String),
    KeyChord {
        ctrl: bool,
        shift: bool,
        alt: bool,
        virtual_key: u16,
    },
}

pub trait InputSink {
    fn send(&mut self, action: &InputAction) -> Result<()>;
}

/// The default backend records events only. This is the mode used in training and whenever the
/// user has not consciously granted permission to inject desktop input.
#[derive(Debug, Default)]
pub struct DryRunSink {
    events: VecDeque<InputAction>,
}

impl DryRunSink {
    pub fn events(&self) -> impl Iterator<Item = &InputAction> {
        self.events.iter()
    }

    pub fn take_events(&mut self) -> Vec<InputAction> {
        self.events.drain(..).collect()
    }
}

impl InputSink for DryRunSink {
    fn send(&mut self, action: &InputAction) -> Result<()> {
        self.events.push_back(action.clone());
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct WindowsInputSink;

impl InputSink for WindowsInputSink {
    fn send(&mut self, action: &InputAction) -> Result<()> {
        #[cfg(windows)]
        {
            windows_input::send(action)
        }
        #[cfg(not(windows))]
        {
            let _ = action;
            Err(anyhow!(
                "EyeOS live desktop input is only available on Windows"
            ))
        }
    }
}

pub struct InputController {
    dry_run: bool,
    dry_run_sink: DryRunSink,
    live_sink: WindowsInputSink,
    recent_events: VecDeque<InputAction>,
}

impl Default for InputController {
    fn default() -> Self {
        Self {
            dry_run: true,
            dry_run_sink: DryRunSink::default(),
            live_sink: WindowsInputSink,
            recent_events: VecDeque::new(),
        }
    }
}

impl InputController {
    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    pub fn set_dry_run(&mut self, dry_run: bool) {
        self.dry_run = dry_run;
    }

    pub fn dispatch(&mut self, action: InputAction) -> Result<()> {
        if self.dry_run {
            self.dry_run_sink.send(&action)?;
        } else {
            self.live_sink.send(&action)?;
        }
        self.recent_events.push_back(action);
        while self.recent_events.len() > 24 {
            self.recent_events.pop_front();
        }
        Ok(())
    }

    pub fn recent_events(&self) -> impl DoubleEndedIterator<Item = &InputAction> {
        self.recent_events.iter()
    }

    pub fn take_dry_run_events(&mut self) -> Vec<InputAction> {
        self.dry_run_sink.take_events()
    }
}

#[cfg(windows)]
mod windows_input {
    use std::mem::size_of;

    use anyhow::{Result, anyhow};
    use windows_sys::Win32::UI::{
        Input::KeyboardAndMouse::{
            INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_KEYUP,
            KEYEVENTF_UNICODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
            MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK,
            MOUSEEVENTF_WHEEL, MOUSEINPUT, SendInput,
        },
        WindowsAndMessaging::{
            GetSystemMetrics, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN,
            SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
        },
    };

    use crate::input::InputAction;

    pub fn send(action: &InputAction) -> Result<()> {
        match action {
            InputAction::MoveTo(point) => mouse_move(point.x, point.y),
            InputAction::LeftClick => send_all(&[
                mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN),
                mouse(0, 0, 0, MOUSEEVENTF_LEFTUP),
            ]),
            InputAction::DoubleClick => send_all(&[
                mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN),
                mouse(0, 0, 0, MOUSEEVENTF_LEFTUP),
                mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN),
                mouse(0, 0, 0, MOUSEEVENTF_LEFTUP),
            ]),
            InputAction::RightClick => send_all(&[
                mouse(0, 0, 0, MOUSEEVENTF_RIGHTDOWN),
                mouse(0, 0, 0, MOUSEEVENTF_RIGHTUP),
            ]),
            InputAction::LeftDown => send_all(&[mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN)]),
            InputAction::LeftUp => send_all(&[mouse(0, 0, 0, MOUSEEVENTF_LEFTUP)]),
            InputAction::ScrollLines(lines) => {
                let amount = lines.saturating_mul(120) as u32;
                send_all(&[mouse(0, 0, amount, MOUSEEVENTF_WHEEL)])
            }
            InputAction::Text(text) => unicode_text(text),
            InputAction::KeyChord {
                ctrl,
                shift,
                alt,
                virtual_key,
            } => key_chord(*ctrl, *shift, *alt, *virtual_key),
        }
    }

    fn mouse_move(x: f64, y: f64) -> Result<()> {
        let left = unsafe { GetSystemMetrics(SM_XVIRTUALSCREEN) };
        let top = unsafe { GetSystemMetrics(SM_YVIRTUALSCREEN) };
        let width = unsafe { GetSystemMetrics(SM_CXVIRTUALSCREEN) }
            .max(unsafe { GetSystemMetrics(SM_CXSCREEN) });
        let height = unsafe { GetSystemMetrics(SM_CYVIRTUALSCREEN) }
            .max(unsafe { GetSystemMetrics(SM_CYSCREEN) });
        let normalized_x = (((x - f64::from(left)) * 65_535.0 / f64::from(width.max(1))).round()
            as i32)
            .clamp(0, 65_535);
        let normalized_y = (((y - f64::from(top)) * 65_535.0 / f64::from(height.max(1))).round()
            as i32)
            .clamp(0, 65_535);
        send_all(&[mouse(
            normalized_x,
            normalized_y,
            0,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        )])
    }

    fn unicode_text(text: &str) -> Result<()> {
        let mut inputs = Vec::with_capacity(text.len() * 2);
        for unit in text.encode_utf16() {
            inputs.push(unicode_keyboard(unit, KEYEVENTF_UNICODE));
            inputs.push(unicode_keyboard(unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
        }
        send_all(&inputs)
    }

    fn key_chord(ctrl: bool, shift: bool, alt: bool, virtual_key: u16) -> Result<()> {
        const VK_CONTROL: u16 = 0x11;
        const VK_SHIFT: u16 = 0x10;
        const VK_MENU: u16 = 0x12;
        let mut inputs = Vec::with_capacity(8);
        for (enabled, key) in [(ctrl, VK_CONTROL), (shift, VK_SHIFT), (alt, VK_MENU)] {
            if enabled {
                inputs.push(virtual_keyboard(key, 0));
            }
        }
        inputs.push(virtual_keyboard(virtual_key, 0));
        inputs.push(virtual_keyboard(virtual_key, KEYEVENTF_KEYUP));
        for (enabled, key) in [(alt, VK_MENU), (shift, VK_SHIFT), (ctrl, VK_CONTROL)] {
            if enabled {
                inputs.push(virtual_keyboard(key, KEYEVENTF_KEYUP));
            }
        }
        send_all(&inputs)
    }

    fn mouse(dx: i32, dy: i32, data: u32, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx,
                    dy,
                    mouseData: data,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn unicode_keyboard(key: u16, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: 0,
                    wScan: key,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn virtual_keyboard(key: u16, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: key,
                    wScan: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn send_all(inputs: &[INPUT]) -> Result<()> {
        let sent = unsafe {
            SendInput(
                inputs.len() as u32,
                inputs.as_ptr(),
                size_of::<INPUT>() as i32,
            )
        };
        if sent != inputs.len() as u32 {
            return Err(anyhow!(
                "Windows rejected injected input ({sent}/{} events sent)",
                inputs.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_records_without_touching_windows() {
        let mut controller = InputController::default();
        controller
            .dispatch(InputAction::Text("hello".to_owned()))
            .unwrap();
        assert_eq!(
            controller.take_dry_run_events(),
            vec![InputAction::Text("hello".to_owned())]
        );
        assert!(controller.is_dry_run());
    }
}
