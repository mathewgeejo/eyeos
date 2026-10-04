//! ABI v1. Handles are opaque IDs owned by the creating thread; calls never run
//! desktop input. Outputs are library-owned byte buffers, released exactly once
//! with et_buffer_free. Borrowed inputs are consumed synchronously.
use crate::{EyeTracker, FrameView, Observation, TrackerConfig};
use serde::Deserialize;
use std::{
    cell::RefCell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub const ET_OK: i32 = 0;
pub const ET_ARGUMENT: i32 = 1;
pub const ET_HANDLE: i32 = 2;
pub const ET_STATE: i32 = 3;
pub const ET_RUNTIME: i32 = 4;
pub const ET_PANIC: i32 = 5;
const MAX_JSON: usize = 4 * 1024 * 1024;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static HANDLES:RefCell<HashMap<u64,EyeTracker>>=RefCell::new(HashMap::new());
    static LAST_ERROR:RefCell<String>=const { RefCell::new(String::new()) };
}
#[repr(C)]
#[derive(Debug, Default)]
pub struct EyeBuffer {
    pub data: *mut u8,
    pub len: usize,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Options {
    #[serde(flatten)]
    tracker: TrackerConfig,
    runtime_root: Option<PathBuf>,
}
type Error = (i32, String);
fn fail(code: i32, message: impl Into<String>) -> Error {
    (code, message.into())
}
fn boundary(f: impl FnOnce() -> Result<(), Error>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(f));
    let (code, message) = match result {
        Ok(Ok(())) => (ET_OK, String::new()),
        Ok(Err(e)) => e,
        Err(_) => (
            ET_PANIC,
            "Rust panic contained at ABI boundary; destroy and recreate this tracker".into(),
        ),
    };
    LAST_ERROR
        .with(|s| *s.borrow_mut() = serde_json::json!({"code":code,"message":message}).to_string());
    code
}
fn with_tracker(
    id: u64,
    f: impl FnOnce(&mut EyeTracker) -> Result<(), Error>,
) -> Result<(), Error> {
    HANDLES.with(|handles| {
        let mut handles = handles.borrow_mut();
        let tracker = handles
            .get_mut(&id)
            .ok_or_else(|| fail(ET_HANDLE, "unknown handle or call from a different thread"))?;
        f(tracker)
    })
}
unsafe fn input<'a>(data: *const u8, len: usize) -> Result<&'a str, Error> {
    if len > MAX_JSON || (data.is_null() && len != 0) {
        return Err(fail(ET_ARGUMENT, "invalid UTF-8 input pointer or length"));
    }
    if len == 0 {
        return Ok("");
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    std::str::from_utf8(bytes).map_err(|_| fail(ET_ARGUMENT, "input must be UTF-8"))
}
unsafe fn output(out: *mut EyeBuffer, value: impl serde::Serialize) -> Result<(), Error> {
    if out.is_null() {
        return Err(fail(ET_ARGUMENT, "output buffer pointer is null"));
    }
    let bytes = serde_json::to_vec(&value)
        .map_err(|e| fail(ET_RUNTIME, e.to_string()))?
        .into_boxed_slice();
    let len = bytes.len();
    let data = Box::into_raw(bytes) as *mut u8;
    unsafe {
        out.write(EyeBuffer { data, len });
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn et_abi_version() -> u32 {
    1
}

/// # Safety
/// Inputs must remain valid for this call; out_handle must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_create(data: *const u8, len: usize, out_handle: *mut u64) -> i32 {
    boundary(|| {
        if out_handle.is_null() {
            return Err(fail(ET_ARGUMENT, "handle output pointer is null"));
        }
        unsafe {
            out_handle.write(0);
        }
        let json = unsafe { input(data, len)? };
        let options: Options = if json.is_empty() {
            Options::default()
        } else {
            serde_json::from_str(json).map_err(|e| fail(ET_ARGUMENT, e.to_string()))?
        };
        let root = options.runtime_root.unwrap_or_else(|| {
            std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join("EyeTracker")
                .join("sdk")
        });
        let tracker = EyeTracker::new(options.tracker, root).map_err(|e| fail(ET_ARGUMENT, e))?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            return Err(fail(ET_RUNTIME, "handle space exhausted"));
        }
        HANDLES.with(|h| h.borrow_mut().insert(id, tracker));
        unsafe {
            out_handle.write(id);
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_destroy(handle: u64) -> i32 {
    boundary(|| {
        HANDLES.with(|h| {
            h.borrow_mut()
                .remove(&handle)
                .map(|_| ())
                .ok_or_else(|| fail(ET_HANDLE, "unknown handle or wrong owning thread"))
        })
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_start_camera(handle: u64, index: u32) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            t.start_camera(index).map_err(|e| fail(ET_RUNTIME, e))
        })
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_stop_camera(handle: u64) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            t.stop_camera();
            Ok(())
        })
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_start_calibration(handle: u64) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            t.start_calibration().map_err(|e| fail(ET_STATE, e))
        })
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_extend_calibration(handle: u64) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            t.extend_calibration().map_err(|e| fail(ET_STATE, e))
        })
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn et_cancel_calibration(handle: u64) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            t.cancel_calibration();
            Ok(())
        })
    })
}
/// # Safety
/// out must be a writable, empty EyeBuffer. Release successful outputs with et_buffer_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_poll(handle: u64, out: *mut EyeBuffer) -> i32 {
    boundary(|| {
        if out.is_null() {
            return Err(fail(ET_ARGUMENT, "output pointer is null"));
        }
        with_tracker(handle, |t| {
            let events = t.poll().map_err(|e| fail(ET_RUNTIME, e))?;
            unsafe { output(out, events) }
        })
    })
}
/// # Safety
/// out must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_timestamp_ms(handle: u64, out: *mut u64) -> i32 {
    boundary(|| {
        if out.is_null() {
            return Err(fail(ET_ARGUMENT, "timestamp output pointer is null"));
        }
        with_tracker(handle, |t| {
            unsafe {
                out.write(t.timestamp_ms());
            }
            Ok(())
        })
    })
}
/// # Safety
/// data must describe a readable RGB24 buffer, and out a writable empty EyeBuffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_process_rgb(
    handle: u64,
    data: *const u8,
    len: usize,
    width: u32,
    height: u32,
    stride: usize,
    timestamp_ms: u64,
    out: *mut EyeBuffer,
) -> i32 {
    boundary(|| {
        if out.is_null() || data.is_null() || len > 8192 * 8192 * 3 {
            return Err(fail(ET_ARGUMENT, "invalid RGB/output pointer or length"));
        }
        // Validate declared dimensions/length before constructing a borrowed slice.
        if width < 2
            || height < 2
            || width > 8192
            || height > 8192
            || stride < width as usize * 3
            || stride.checked_mul(height as usize).is_none_or(|n| n > len)
        {
            return Err(fail(
                ET_ARGUMENT,
                "invalid RGB dimensions, stride, or buffer length",
            ));
        }
        with_tracker(handle, |t| {
            let frame = FrameView {
                rgb: unsafe { std::slice::from_raw_parts(data, len) },
                width,
                height,
                stride_bytes: stride,
                timestamp_ms,
            };
            let event = t.process_frame(frame).map_err(|e| fail(ET_RUNTIME, e))?;
            unsafe { output(out, event) }
        })
    })
}
/// Integration seam for custom vision adapters and deterministic replay. Labels
/// are still supplied exclusively by the calibration session's displayed targets.
/// # Safety
/// data is valid UTF-8 JSON and out is a writable empty EyeBuffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_process_observation(
    handle: u64,
    data: *const u8,
    len: usize,
    out: *mut EyeBuffer,
) -> i32 {
    boundary(|| {
        if out.is_null() {
            return Err(fail(ET_ARGUMENT, "output pointer is null"));
        }
        let observation: Observation = serde_json::from_str(unsafe { input(data, len)? })
            .map_err(|e| fail(ET_ARGUMENT, e.to_string()))?;
        with_tracker(handle, |t| {
            let now = t.timestamp_ms();
            let event = t.observe(observation, now).map_err(|e| fail(ET_STATE, e))?;
            unsafe { output(out, event) }
        })
    })
}
/// # Safety
/// out is a writable empty EyeBuffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_calibration_progress(handle: u64, out: *mut EyeBuffer) -> i32 {
    boundary(|| with_tracker(handle, |t| unsafe { output(out, t.calibration_progress()) }))
}
/// # Safety
/// out is a writable empty EyeBuffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_export_profile(handle: u64, out: *mut EyeBuffer) -> i32 {
    boundary(|| {
        with_tracker(handle, |t| {
            let profile = t
                .engine()
                .profile()
                .ok_or_else(|| fail(ET_STATE, "no calibration profile"))?;
            unsafe { output(out, profile) }
        })
    })
}
/// # Safety
/// data is readable UTF-8 for len bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_import_profile(handle: u64, data: *const u8, len: usize) -> i32 {
    boundary(|| {
        let json = unsafe { input(data, len)? };
        with_tracker(handle, |t| {
            t.engine_mut()
                .import_profile(json)
                .map_err(|e| fail(ET_STATE, e))
        })
    })
}
/// # Safety
/// out is a writable empty EyeBuffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_last_error(out: *mut EyeBuffer) -> i32 {
    // Do not clear the error while reading it.
    let result = catch_unwind(AssertUnwindSafe(|| {
        LAST_ERROR.with(|e| {
            let value: serde_json::Value = serde_json::from_str(&e.borrow())
                .unwrap_or_else(|_| serde_json::json!({"code":0,"message":""}));
            unsafe { output(out, value) }
        })
    }));
    match result {
        Ok(Ok(())) => ET_OK,
        Ok(Err((code, _))) => code,
        Err(_) => ET_PANIC,
    }
}
/// # Safety
/// buffer is a live allocation returned by this library, freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn et_buffer_free(buffer: *mut EyeBuffer) -> i32 {
    boundary(|| {
        if buffer.is_null() {
            return Err(fail(ET_ARGUMENT, "buffer pointer is null"));
        }
        let buffer = unsafe { &mut *buffer };
        if !buffer.data.is_null() {
            let slice = std::ptr::slice_from_raw_parts_mut(buffer.data, buffer.len);
            unsafe {
                drop(Box::from_raw(slice));
            }
        } else if buffer.len != 0 {
            return Err(fail(ET_ARGUMENT, "null buffer has nonzero length"));
        }
        *buffer = EyeBuffer::default();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handles_and_owned_buffers_have_explicit_lifetimes() {
        let mut handle = 0;
        assert_eq!(
            unsafe { et_create(std::ptr::null(), 0, &mut handle) },
            ET_OK
        );
        let mut buffer = EyeBuffer::default();
        assert_eq!(unsafe { et_poll(handle, &mut buffer) }, ET_OK);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(buffer.data, buffer.len) },
            b"[]"
        );
        assert_eq!(unsafe { et_buffer_free(&mut buffer) }, ET_OK);
        assert!(buffer.data.is_null());
        assert_eq!(et_destroy(handle), ET_OK);
        assert_eq!(et_destroy(handle), ET_HANDLE);
    }
    #[test]
    fn null_invalid_json_and_wrong_thread_return_structured_errors() {
        assert_eq!(
            unsafe { et_create(std::ptr::null(), 4, std::ptr::null_mut()) },
            ET_ARGUMENT
        );
        let mut handle = 0;
        assert_eq!(
            unsafe { et_create(b"{".as_ptr(), 1, &mut handle) },
            ET_ARGUMENT
        );
        assert_eq!(
            unsafe { et_create(std::ptr::null(), 0, &mut handle) },
            ET_OK
        );
        assert_eq!(
            std::thread::spawn(move || et_start_calibration(handle))
                .join()
                .unwrap(),
            ET_HANDLE
        );
        assert_eq!(et_destroy(handle), ET_OK);
    }
    #[test]
    fn panics_are_contained() {
        assert_eq!(boundary(|| panic!("intentional boundary test")), ET_PANIC);
    }
}
