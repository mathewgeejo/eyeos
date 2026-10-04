//! Windows-only local webcam tracker backed by the MediaPipe Face Landmarker C API.
//!
//! The worker owns the webcam and native runtimes. It retains each RGB frame only long enough to
//! crop the face and eyes, then emits the local OpenVINO gaze-vector feature. Frames, landmarks,
//! and model output never leave this process or get persisted.

use std::{
    collections::VecDeque,
    ffi::{CStr, c_char, c_void},
    fs,
    path::{Path, PathBuf},
    ptr::{null, null_mut},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use sha2::{Digest, Sha256};

use crate::{
    gaze_estimator::GazeEstimator,
    vision::{
        EyeFeatures, LANDMARK_COUNT, Landmark, LandmarkFrame, embedded_face_landmarker_model,
    },
};

const MEDIAPIPE_RUNTIME_SHA256: &str =
    "aa8e6c1b618c30cd3a6ad584dee1b2f2c99c3f3025d683bada36e1566d9092b7";
const MEDIAPIPE_RUNTIME: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/runtime/mediapipe/libmediapipe.dll"
));

#[derive(Debug, Clone, serde::Serialize)]
pub enum TrackerStatus {
    Starting,
    CameraReady {
        width: u32,
        height: u32,
        fps: u32,
        format: String,
    },
    CameraRetrying {
        attempt: u32,
        detail: String,
    },
    Tracking {
        fps: f32,
    },
    LowFrameRate {
        fps: f32,
    },
    GazeUnavailable {
        detail: String,
    },
    NoFace,
    Failed(String),
    Stopped,
}

#[derive(Debug, Clone)]
pub enum TrackerEvent {
    Features {
        features: EyeFeatures,
        timestamp_ms: u64,
    },
    Status(TrackerStatus),
}

/// Owns the communication channels for the local capture/inference worker.
#[derive(Clone)]
pub(crate) struct EventSender {
    shared: Arc<EventMailbox>,
}
struct EventMailbox {
    pending: Mutex<VecDeque<TrackerEvent>>,
    closed: AtomicBool,
}
impl EventSender {
    pub(crate) fn send(&self, event: TrackerEvent) -> Result<(), ()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(());
        }
        let mut pending = self.shared.pending.lock().map_err(|_| ())?;
        let failure = matches!(
            &event,
            TrackerEvent::Status(
                TrackerStatus::NoFace
                    | TrackerStatus::GazeUnavailable { .. }
                    | TrackerStatus::CameraRetrying { .. }
                    | TrackerStatus::Failed(_)
                    | TrackerStatus::Stopped
            )
        );
        if failure || matches!(&event, TrackerEvent::Features { .. }) {
            pending.retain(|e| !matches!(e, TrackerEvent::Features { .. }));
        }
        if pending.len() >= 16 {
            pending.pop_front();
        }
        pending.push_back(event);
        Ok(())
    }
}

pub struct LocalTracker {
    events: Arc<EventMailbox>,
    stop: Sender<()>,
}

impl LocalTracker {
    pub fn start(profile_root: PathBuf, camera_index: u32) -> Result<Self> {
        Self::start_at(profile_root, camera_index, Instant::now())
    }
    pub fn start_at(profile_root: PathBuf, camera_index: u32, clock: Instant) -> Result<Self> {
        #[cfg(feature = "camera")]
        {
            let runtime_path = extract_runtime(&profile_root)?;
            let events = Arc::new(EventMailbox {
                pending: Mutex::new(VecDeque::new()),
                closed: AtomicBool::new(false),
            });
            let event_tx = EventSender {
                shared: events.clone(),
            };
            let (stop, stop_rx) = mpsc::channel();
            thread::Builder::new()
                .name("eyeos-face-tracker".to_owned())
                .spawn(move || {
                    run_worker(
                        runtime_path,
                        profile_root,
                        camera_index,
                        event_tx,
                        stop_rx,
                        clock,
                    )
                })
                .context("starting the local eye-tracking worker")?;
            Ok(Self { events, stop })
        }
        #[cfg(not(feature = "camera"))]
        {
            let _ = (profile_root, camera_index, clock);
            Err(anyhow!("camera support was excluded at build time"))
        }
    }

    pub fn drain(&self) -> Vec<TrackerEvent> {
        self.events
            .pending
            .lock()
            .map(|mut events| events.drain(..).collect())
            .unwrap_or_default()
    }
}

impl Drop for LocalTracker {
    fn drop(&mut self) {
        self.events.closed.store(true, Ordering::Release);
        let _ = self.stop.send(());
    }
}

pub(crate) fn extract_runtime(profile_root: &Path) -> Result<PathBuf> {
    if sha256_hex(MEDIAPIPE_RUNTIME) != MEDIAPIPE_RUNTIME_SHA256 {
        bail!("the embedded MediaPipe runtime did not match its pinned SHA-256")
    }
    let runtime_directory = profile_root.join("runtime").join("mediapipe");
    fs::create_dir_all(&runtime_directory)
        .with_context(|| format!("creating {}", runtime_directory.display()))?;
    let runtime_path = runtime_directory.join(format!(
        "libmediapipe-{}.dll",
        &MEDIAPIPE_RUNTIME_SHA256[..12]
    ));
    if runtime_path.exists() {
        let existing = fs::read(&runtime_path)
            .with_context(|| format!("reading {}", runtime_path.display()))?;
        if sha256_hex(&existing) == MEDIAPIPE_RUNTIME_SHA256 {
            return Ok(runtime_path);
        }
        bail!(
            "the existing MediaPipe runtime at {} has an unexpected hash; remove only that file and restart EyeOS",
            runtime_path.display()
        );
    }
    fs::write(&runtime_path, MEDIAPIPE_RUNTIME)
        .with_context(|| format!("writing {}", runtime_path.display()))?;
    Ok(runtime_path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(feature = "camera")]
fn run_worker(
    runtime_path: PathBuf,
    profile_root: PathBuf,
    camera_index: u32,
    event_tx: EventSender,
    stop_rx: Receiver<()>,
    clock: Instant,
) {
    let _ = event_tx.send(TrackerEvent::Status(TrackerStatus::Starting));
    let result = (|| {
        let mut gaze_estimator = GazeEstimator::load(&profile_root)
            .context("loading the bundled OpenVINO gaze-estimation pipeline")?;
        run_worker_inner(
            &runtime_path,
            &mut gaze_estimator,
            camera_index,
            &event_tx,
            &stop_rx,
            clock,
        )
    })();
    match result {
        Ok(()) => {
            let _ = event_tx.send(TrackerEvent::Status(TrackerStatus::Stopped));
        }
        Err(error) => {
            eprintln!("[EyeOS tracker] {error:#}");
            let _ = event_tx.send(TrackerEvent::Status(TrackerStatus::Failed(
                error.to_string(),
            )));
        }
    }
}

#[cfg(feature = "camera")]
fn run_worker_inner(
    runtime_path: &Path,
    gaze_estimator: &mut GazeEstimator,
    camera_index: u32,
    event_tx: &EventSender,
    stop_rx: &Receiver<()>,
    clock: Instant,
) -> Result<()> {
    use nokhwa::{
        pixel_format::RgbFormat,
        utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    };

    let landmarker = MediaPipeFaceLandmarker::load(runtime_path)?;
    let mut attempt = 0_u32;
    loop {
        attempt = attempt.saturating_add(1);
        let requested = if attempt % 2 == 1 {
            // Do not force MJPEG. Integrated webcams commonly expose only NV12 or YUYV.
            // RgbFormat accepts all of those formats and asks Windows for the best 30-FPS
            // option, which is the practical baseline for responsive gaze control.
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(30))
        } else {
            // A few webcams do not publish an exact 30-FPS mode. On retry, accept their
            // fastest RGB-decodable format rather than treating a detected camera as usable.
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate)
        };

        match run_capture_session(
            &landmarker,
            gaze_estimator,
            CameraIndex::Index(camera_index),
            requested,
            event_tx,
            stop_rx,
            clock,
        ) {
            Ok(()) => return Ok(()),
            Err(error) => {
                let detail = format!("Camera attempt {attempt} failed: {error:#}");
                eprintln!("[EyeOS tracker] {detail}");
                let _ = event_tx.send(TrackerEvent::Status(TrackerStatus::CameraRetrying {
                    attempt,
                    detail,
                }));
                match stop_rx.recv_timeout(Duration::from_millis(900)) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        }
    }
}

/// Opens one capture session. A transient camera read failure ends this session so the owning
/// worker can fully release Media Foundation and reopen it with a fresh format negotiation.
#[cfg(feature = "camera")]
fn run_capture_session(
    landmarker: &MediaPipeFaceLandmarker,
    gaze_estimator: &mut GazeEstimator,
    camera_index: nokhwa::utils::CameraIndex,
    requested: nokhwa::utils::RequestedFormat<'static>,
    event_tx: &EventSender,
    stop_rx: &Receiver<()>,
    clock: Instant,
) -> Result<()> {
    crate::capture::run(
        landmarker,
        gaze_estimator,
        camera_index,
        requested,
        event_tx,
        stop_rx,
        clock,
    )
}

#[cfg(feature = "native")]
type MpFaceLandmarkerPtr = *mut c_void;
#[cfg(feature = "native")]
type MpImagePtr = *mut c_void;

#[cfg(feature = "native")]
#[repr(C)]
struct BaseOptions {
    model_asset_buffer: *const c_char,
    model_asset_buffer_count: u32,
    model_asset_path: *const c_char,
    delegate: i32,
    host_environment: i32,
    host_system: i32,
    host_version: *const c_char,
    ca_bundle_path: *const c_char,
}

#[cfg(feature = "native")]
#[repr(C)]
struct FaceLandmarkerOptions {
    base_options: BaseOptions,
    running_mode: i32,
    num_faces: i32,
    min_face_detection_confidence: f32,
    min_face_presence_confidence: f32,
    min_tracking_confidence: f32,
    output_face_blendshapes: bool,
    output_facial_transformation_matrixes: bool,
    result_callback:
        Option<unsafe extern "C" fn(i32, *const FaceLandmarkerResult, MpImagePtr, i64)>,
}

#[cfg(feature = "native")]
#[repr(C)]
struct NormalizedLandmark {
    x: f32,
    y: f32,
    z: f32,
    has_visibility: bool,
    visibility: f32,
    has_presence: bool,
    presence: f32,
    name: *mut c_char,
}

#[cfg(feature = "native")]
#[repr(C)]
struct NormalizedLandmarks {
    landmarks: *mut NormalizedLandmark,
    landmarks_count: u32,
}

#[cfg(feature = "native")]
#[repr(C)]
struct FaceLandmarkerResult {
    face_landmarks: *mut NormalizedLandmarks,
    face_landmarks_count: u32,
    face_blendshapes: *mut c_void,
    face_blendshapes_count: u32,
    facial_transformation_matrixes: *mut c_void,
    facial_transformation_matrixes_count: u32,
}

#[cfg(feature = "native")]
type CreateLandmarker = unsafe extern "C" fn(
    *mut FaceLandmarkerOptions,
    *mut MpFaceLandmarkerPtr,
    *mut *mut c_char,
) -> i32;
#[cfg(feature = "native")]
type DetectForVideo = unsafe extern "C" fn(
    MpFaceLandmarkerPtr,
    MpImagePtr,
    *const c_void,
    i64,
    *mut FaceLandmarkerResult,
    *mut *mut c_char,
) -> i32;
#[cfg(feature = "native")]
type CloseResult = unsafe extern "C" fn(*mut FaceLandmarkerResult);
#[cfg(feature = "native")]
type CloseLandmarker = unsafe extern "C" fn(MpFaceLandmarkerPtr, *mut *mut c_char) -> i32;
#[cfg(feature = "native")]
type CreateImage =
    unsafe extern "C" fn(i32, i32, i32, *const u8, i32, *mut MpImagePtr, *mut *mut c_char) -> i32;
#[cfg(feature = "native")]
type FreeImage = unsafe extern "C" fn(MpImagePtr);
#[cfg(feature = "native")]
type FreeError = unsafe extern "C" fn(*mut c_char);

#[cfg(feature = "native")]
pub(crate) struct MediaPipeFaceLandmarker {
    _library: Library,
    create_landmarker: CreateLandmarker,
    detect_for_video: DetectForVideo,
    close_result: CloseResult,
    close_landmarker: CloseLandmarker,
    create_image: CreateImage,
    free_image: FreeImage,
    free_error: FreeError,
    handle: MpFaceLandmarkerPtr,
}

#[cfg(feature = "native")]
impl MediaPipeFaceLandmarker {
    pub(crate) fn load(runtime_path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(runtime_path) }
            .with_context(|| format!("loading {}", runtime_path.display()))?;
        let create_landmarker = unsafe { load_symbol(&library, b"MpFaceLandmarkerCreate\0")? };
        let detect_for_video =
            unsafe { load_symbol(&library, b"MpFaceLandmarkerDetectForVideo\0")? };
        let close_result = unsafe { load_symbol(&library, b"MpFaceLandmarkerCloseResult\0")? };
        let close_landmarker = unsafe { load_symbol(&library, b"MpFaceLandmarkerClose\0")? };
        let create_image = unsafe { load_symbol(&library, b"MpImageCreateFromUint8Data\0")? };
        let free_image = unsafe { load_symbol(&library, b"MpImageFree\0")? };
        let free_error = unsafe { load_symbol(&library, b"MpErrorFree\0")? };

        let mut task = Self {
            _library: library,
            create_landmarker,
            detect_for_video,
            close_result,
            close_landmarker,
            create_image,
            free_image,
            free_error,
            handle: null_mut(),
        };
        task.create()?;
        Ok(task)
    }

    fn create(&mut self) -> Result<()> {
        const EYEOS_VERSION: &[u8] = b"EyeOS/0.1.0\0";
        let model = embedded_face_landmarker_model();
        let mut options = FaceLandmarkerOptions {
            base_options: BaseOptions {
                model_asset_buffer: model.as_ptr().cast(),
                model_asset_buffer_count: model
                    .len()
                    .try_into()
                    .map_err(|_| anyhow!("the embedded MediaPipe model is too large"))?,
                model_asset_path: null(),
                delegate: 0,
                host_environment: 0,
                host_system: 3,
                host_version: EYEOS_VERSION.as_ptr().cast(),
                ca_bundle_path: null(),
            },
            running_mode: 2,
            num_faces: 1,
            min_face_detection_confidence: 0.70,
            min_face_presence_confidence: 0.70,
            min_tracking_confidence: 0.70,
            output_face_blendshapes: false,
            output_facial_transformation_matrixes: false,
            result_callback: None,
        };
        let mut error = null_mut();
        let status =
            unsafe { (self.create_landmarker)(&mut options, &mut self.handle, &mut error) };
        self.status_result(status, error, "creating the MediaPipe face landmarker")?;
        if self.handle.is_null() {
            bail!("MediaPipe reported success but returned no face-landmarker handle")
        }
        Ok(())
    }

    pub(crate) fn detect_rgb(
        &self,
        rgb: &[u8],
        width: i32,
        height: i32,
        timestamp_ms: u64,
    ) -> Result<Option<LandmarkFrame>> {
        let mut image = null_mut();
        let mut error = null_mut();
        let status = unsafe {
            (self.create_image)(
                1,
                width,
                height,
                rgb.as_ptr(),
                rgb.len()
                    .try_into()
                    .map_err(|_| anyhow!("camera frame is too large"))?,
                &mut image,
                &mut error,
            )
        };
        self.status_result(status, error, "creating a MediaPipe RGB image")?;
        if image.is_null() {
            bail!("MediaPipe reported success but returned no image handle")
        }

        let mut result: FaceLandmarkerResult = unsafe { std::mem::zeroed() };
        let mut error = null_mut();
        let status = unsafe {
            (self.detect_for_video)(
                self.handle,
                image,
                null(),
                timestamp_ms as i64,
                &mut result,
                &mut error,
            )
        };
        unsafe { (self.free_image)(image) };
        self.status_result(status, error, "running MediaPipe face landmark inference")?;

        let frame = self.copy_first_face(&result, timestamp_ms);
        unsafe { (self.close_result)(&mut result) };
        frame
    }

    fn copy_first_face(
        &self,
        result: &FaceLandmarkerResult,
        timestamp_ms: u64,
    ) -> Result<Option<LandmarkFrame>> {
        if result.face_landmarks_count == 0 {
            return Ok(None);
        }
        if result.face_landmarks.is_null() {
            bail!("MediaPipe returned a face count without landmark data")
        }
        let landmarks = unsafe { &*result.face_landmarks };
        if landmarks.landmarks.is_null() || landmarks.landmarks_count < LANDMARK_COUNT as u32 {
            bail!(
                "MediaPipe returned {} landmarks; EyeOS requires all {LANDMARK_COUNT} face/iris landmarks",
                landmarks.landmarks_count
            )
        }
        let raw = unsafe { std::slice::from_raw_parts(landmarks.landmarks, LANDMARK_COUNT) };
        let landmarks = raw
            .iter()
            .map(|landmark| Landmark {
                x: landmark.x,
                y: landmark.y,
                z: landmark.z,
                confidence: if landmark.has_presence {
                    landmark.presence.clamp(0.0, 1.0)
                } else {
                    f32::NAN
                },
            })
            .collect();
        Ok(Some(LandmarkFrame {
            landmarks,
            timestamp_ms,
            // Unknown confidence is preserved; task thresholds are not a measured score.
            face_confidence: f32::NAN,
        }))
    }

    fn status_result(&self, status: i32, error: *mut c_char, operation: &str) -> Result<()> {
        if status == 0 {
            if !error.is_null() {
                unsafe { (self.free_error)(error) };
            }
            return Ok(());
        }
        let message = if error.is_null() {
            format!("MediaPipe status {status}")
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { (self.free_error)(error) };
            message
        };
        Err(anyhow!("{operation}: {message}"))
    }
}

#[cfg(feature = "native")]
impl Drop for MediaPipeFaceLandmarker {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let mut error = null_mut();
        let _ = unsafe { (self.close_landmarker)(self.handle, &mut error) };
        if !error.is_null() {
            unsafe { (self.free_error)(error) };
        }
        self.handle = null_mut();
    }
}

#[cfg(feature = "native")]
unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    let symbol = unsafe { library.get::<T>(name) }
        .with_context(|| format!("resolving MediaPipe export {:?}", name))?;
    Ok(*symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_runtime_matches_its_pinned_hash() {
        assert_eq!(sha256_hex(MEDIAPIPE_RUNTIME), MEDIAPIPE_RUNTIME_SHA256);
    }

    #[cfg(feature = "native")]
    #[test]
    fn ffi_layout_matches_the_pinned_mediapipe_headers_on_x64() {
        use std::mem::size_of;
        assert_eq!(size_of::<BaseOptions>(), 56);
        assert_eq!(size_of::<FaceLandmarkerOptions>(), 88);
        assert_eq!(size_of::<NormalizedLandmark>(), 40);
        assert_eq!(size_of::<NormalizedLandmarks>(), 16);
        assert_eq!(size_of::<FaceLandmarkerResult>(), 48);
    }

    #[cfg(all(feature = "camera", windows))]
    #[test]
    fn embedded_runtime_creates_the_pinned_face_landmarker() {
        let directory = tempfile::tempdir().expect("temporary runtime directory");
        let runtime_path = extract_runtime(directory.path()).expect("extract runtime");
        if let Err(error) = MediaPipeFaceLandmarker::load(&runtime_path) {
            panic!("MediaPipe C API did not initialise: {error:#}");
        }
    }
}
