//! Capture and inference run separately. A single replaceable frame slot bounds
//! memory and prevents inference from processing a growing camera backlog.
use crate::{
    gaze_estimator::GazeEstimator,
    tracker::{EventSender, MediaPipeFaceLandmarker, TrackerEvent, TrackerStatus},
};
use anyhow::{Result, anyhow};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat},
};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

struct Captured {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    timestamp_ms: u64,
}
type FrameSlot = Arc<(Mutex<Option<Result<Captured, String>>>, Condvar)>;

fn publish(slot: &FrameSlot, frame: Result<Captured, String>) {
    if let Ok(mut value) = slot.0.lock() {
        *value = Some(frame);
        slot.1.notify_one();
    }
}
fn capture_loop(
    index: CameraIndex,
    format: RequestedFormat<'static>,
    events: EventSender,
    slot: FrameSlot,
    stopped: Arc<AtomicBool>,
    clock: Instant,
) -> Result<()> {
    let mut camera = Camera::new(index, format).map_err(|e| anyhow!("opening webcam: {e}"))?;
    camera
        .open_stream()
        .map_err(|e| anyhow!("opening camera stream: {e}"))?;
    let size = camera.resolution();
    let _ = events.send(TrackerEvent::Status(TrackerStatus::CameraReady {
        width: size.width_x,
        height: size.height_y,
        fps: camera.frame_rate(),
        format: camera.frame_format().to_string(),
    }));
    let mut errors = 0;
    while !stopped.load(Ordering::Acquire) {
        match camera.frame() {
            Ok(frame) => {
                errors = 0;
                // Timestamp before decoding and inference, using the session's shared clock.
                let timestamp_ms = clock.elapsed().as_millis() as u64;
                let size = frame.resolution();
                let rgb = frame
                    .decode_image::<RgbFormat>()
                    .map_err(|e| anyhow!("RGB decoding: {e}"))?;
                publish(
                    &slot,
                    Ok(Captured {
                        rgb: rgb.into_raw(),
                        width: size.width_x,
                        height: size.height_y,
                        timestamp_ms,
                    }),
                );
            }
            Err(error) => {
                errors += 1;
                let _ = events.send(TrackerEvent::Status(TrackerStatus::GazeUnavailable {
                    detail: format!("camera read failed: {error}"),
                }));
                if errors >= 5 {
                    return Err(anyhow!("camera stopped providing frames: {error}"));
                }
                thread::sleep(Duration::from_millis(30));
            }
        }
    }
    let _ = camera.stop_stream();
    Ok(())
}

pub(crate) fn run(
    landmarker: &MediaPipeFaceLandmarker,
    estimator: &mut GazeEstimator,
    index: CameraIndex,
    format: RequestedFormat<'static>,
    events: &EventSender,
    stop: &Receiver<()>,
    clock: Instant,
) -> Result<()> {
    let slot: FrameSlot = Arc::new((Mutex::new(None), Condvar::new()));
    let stopped = Arc::new(AtomicBool::new(false));
    let capture_slot = slot.clone();
    let capture_stopped = stopped.clone();
    let capture_events = events.clone();
    let producer = thread::Builder::new()
        .name("eye-tracker-capture".into())
        .spawn(move || {
            if let Err(error) = capture_loop(
                index,
                format,
                capture_events,
                capture_slot.clone(),
                capture_stopped,
                clock,
            ) {
                publish(&capture_slot, Err(error.to_string()));
            }
        })?;
    let mut measured = Instant::now();
    let mut frames = 0;
    let result = (|| -> Result<()> {
        loop {
            match stop.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => return Ok(()),
                Err(TryRecvError::Empty) => {}
            }
            let guard = slot
                .0
                .lock()
                .map_err(|_| anyhow!("capture frame slot poisoned"))?;
            let (mut guard, _) = slot
                .1
                .wait_timeout_while(guard, Duration::from_millis(40), |v| v.is_none())
                .map_err(|_| anyhow!("capture frame slot poisoned"))?;
            let Some(packet) = guard.take() else {
                continue;
            };
            drop(guard);
            let frame = packet.map_err(|e| anyhow!(e))?;
            if clock.elapsed().as_millis() as u64 - frame.timestamp_ms > 200 {
                continue;
            }
            let features = match landmarker.detect_rgb(
                &frame.rgb,
                frame.width as i32,
                frame.height as i32,
                frame.timestamp_ms,
            )? {
                Some(landmarks) => {
                    match estimator.estimate(&frame.rgb, frame.width, frame.height, &landmarks) {
                        Ok(mut features) => {
                            features.inference_latency_ms =
                                (clock.elapsed().as_millis() as u64 - frame.timestamp_ms) as f64;
                            Some(features)
                        }
                        Err(error) => {
                            let _ =
                                events.send(TrackerEvent::Status(TrackerStatus::GazeUnavailable {
                                    detail: error.to_string(),
                                }));
                            None
                        }
                    }
                }
                None => {
                    let _ = events.send(TrackerEvent::Status(TrackerStatus::NoFace));
                    None
                }
            };
            if let Some(features) = features {
                if events
                    .send(TrackerEvent::Features {
                        features,
                        timestamp_ms: frame.timestamp_ms,
                    })
                    .is_err()
                {
                    return Ok(());
                }
            }
            frames += 1;
            if measured.elapsed() >= Duration::from_secs(1) {
                let fps = frames as f32 / measured.elapsed().as_secs_f32();
                frames = 0;
                measured = Instant::now();
                let _ = events.send(TrackerEvent::Status(if fps >= 25.0 {
                    TrackerStatus::Tracking { fps }
                } else {
                    TrackerStatus::LowFrameRate { fps }
                }));
            }
        }
    })();
    stopped.store(true, Ordering::Release);
    // Camera APIs can block in a driver; do not hang the host's UI on shutdown.
    // On normal operation the producer sees this flag at the next frame and closes.
    if producer.is_finished() {
        let _ = producer.join();
    }
    result
}
