//! Local OpenVINO gaze-estimation pipeline.
//!
//! MediaPipe is intentionally used here only for face/eye geometry.  The actual gaze feature
//! comes from the bundled Open Model Zoo head-pose and gaze-vector networks, both executed on
//! the local CPU.  Frames and inference results remain in this process and are never persisted.

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Shape, Tensor};
use sha2::{Digest, Sha256};

use crate::vision::{EyeFeatures, LANDMARK_COUNT, Landmark, LandmarkFrame};
use eye_tracker_core::{Point, Quality, normalize_gaze_vector};

const IMAGE_SIDE: usize = 60;
const IMAGE_VALUES: usize = 3 * IMAGE_SIDE * IMAGE_SIDE;

const LEFT_EYE_OUTER: usize = 33;
const LEFT_EYE_INNER: usize = 133;
const RIGHT_EYE_INNER: usize = 362;
const RIGHT_EYE_OUTER: usize = 263;

const GAZE_XML: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/openvino/gaze-estimation-adas-0002.xml"
));
const GAZE_BIN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/openvino/gaze-estimation-adas-0002.bin"
));
const HEAD_POSE_XML: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/openvino/head-pose-estimation-adas-0001.xml"
));
const HEAD_POSE_BIN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/openvino/head-pose-estimation-adas-0001.bin"
));

const OPENVINO_DLLS: &[EmbeddedAsset] = &[
    EmbeddedAsset::new(
        "openvino.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino.dll"
        )),
        "a2e71c1885c01a3aa11e378bd8d09ba9b3e16e47460582a4250b216bb5446da3",
    ),
    EmbeddedAsset::new(
        "openvino_c.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino_c.dll"
        )),
        "9a3610bdd1ee59c01f799672ed714f93ebb9fcfca0d885b4662f885cff830932",
    ),
    EmbeddedAsset::new(
        "openvino_intel_cpu_plugin.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino_intel_cpu_plugin.dll"
        )),
        "17f23dd72f0ccdbb192f98fdbf9b45c32bbab0ac12d7f0aa9541eaffb52117a5",
    ),
    EmbeddedAsset::new(
        "openvino_ir_frontend.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino_ir_frontend.dll"
        )),
        "28332b83c2162c46866592faae78bd1f13cfa3c144efc1264d2ea22f29963d25",
    ),
    EmbeddedAsset::new(
        "openvino_auto_plugin.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino_auto_plugin.dll"
        )),
        "d96f895ef07caf3e206b42b16b66f25b41f1d44d79a07c7d0acae3d713d72faa",
    ),
    EmbeddedAsset::new(
        "openvino_hetero_plugin.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/openvino_hetero_plugin.dll"
        )),
        "a3a55fb2216e51b8fc0a6314ff19bb407bca688214c2ed88135a8d42a0ac8e2e",
    ),
    EmbeddedAsset::new(
        "tbb12.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/tbb12.dll"
        )),
        "311e8293524a154786f55f729b61e4f9ffa7d24f6b612aba52fa706f8c8c60b6",
    ),
    EmbeddedAsset::new(
        "tbbbind_2_5.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/tbbbind_2_5.dll"
        )),
        "0472419da5de3f2001ad61e321e0c48b1f50038b5693e3597543f50eadcada17",
    ),
    EmbeddedAsset::new(
        "tbbmalloc.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/tbbmalloc.dll"
        )),
        "dfce1b1c275b0549714b5c6b1410577e41bd12d6e0c772c0894c1f270a6cee01",
    ),
    EmbeddedAsset::new(
        "tbbmalloc_proxy.dll",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/runtime/openvino/tbbmalloc_proxy.dll"
        )),
        "31e4064aa16b749c5aac9a8f8fa7c8f8ae86079328b3ab7a9947e20ebae06c7f",
    ),
];

const MODEL_ASSETS: &[EmbeddedAsset] = &[
    EmbeddedAsset::new(
        "gaze-estimation-adas-0002.xml",
        GAZE_XML,
        "b0648e6f6efa1437f13e5a4c137e7824a9aa8e80c6b93e82e592ce3f1b8eef43",
    ),
    EmbeddedAsset::new(
        "gaze-estimation-adas-0002.bin",
        GAZE_BIN,
        "dfa42c854ec43e0da6aacf34ea6729019edebd39124b6897323488d6fe8fb01e",
    ),
    EmbeddedAsset::new(
        "head-pose-estimation-adas-0001.xml",
        HEAD_POSE_XML,
        "2e159342a5d17fab5b8018158e8dfc60ced89a0f651a2421e765c8fa0594b033",
    ),
    EmbeddedAsset::new(
        "head-pose-estimation-adas-0001.bin",
        HEAD_POSE_BIN,
        "a4e66d59e8053681b0c20ef9ee519583d6925e4f6c5002005f36403e414b133e",
    ),
];

struct EmbeddedAsset {
    name: &'static str,
    bytes: &'static [u8],
    sha256: &'static str,
}

impl EmbeddedAsset {
    const fn new(name: &'static str, bytes: &'static [u8], sha256: &'static str) -> Self {
        Self {
            name,
            bytes,
            sha256,
        }
    }
}

/// A compiled, reusable CPU gaze estimator.  It owns the OpenVINO requests and input tensors so
/// a frame does not allocate an inference request or recompile either network.
pub struct GazeEstimator {
    head_request: InferRequest,
    gaze_request: InferRequest,
    // Requests must be dropped before their compiled models, and compiled models before Core.
    _head_model: CompiledModel,
    _gaze_model: CompiledModel,
    _core: Core,
    head_input: Tensor,
    left_eye_input: Tensor,
    right_eye_input: Tensor,
    head_angles_input: Tensor,
    // Hold dependencies until after all OpenVINO handles have been dropped.
    _runtime_dependencies: Vec<libloading::os::windows::Library>,
}

impl GazeEstimator {
    /// Extract verified local assets and compile both networks for OpenVINO's CPU device.
    pub fn load(profile_root: &Path) -> Result<Self> {
        let runtime_directory = profile_root.join("runtime").join("openvino");
        extract_assets(&runtime_directory, OPENVINO_DLLS)?;
        let model_directory = profile_root.join("models").join("openvino");
        extract_assets(&model_directory, MODEL_ASSETS)?;

        let runtime_dependencies = load_runtime_dependencies(&runtime_directory)?;
        openvino_sys::library::load_from(runtime_directory.join("openvino_c.dll"))
            .map_err(|error| anyhow!("loading bundled OpenVINO runtime: {error}"))?;

        let mut core = Core::new().context("creating the local OpenVINO CPU runtime")?;
        let head_xml = model_directory.join("head-pose-estimation-adas-0001.xml");
        let head_bin = model_directory.join("head-pose-estimation-adas-0001.bin");
        let gaze_xml = model_directory.join("gaze-estimation-adas-0002.xml");
        let gaze_bin = model_directory.join("gaze-estimation-adas-0002.bin");

        let head_network = core
            .read_model_from_file(&display_path(&head_xml), &display_path(&head_bin))
            .context("reading the embedded head-pose model")?;
        let gaze_network = core
            .read_model_from_file(&display_path(&gaze_xml), &display_path(&gaze_bin))
            .context("reading the embedded gaze-vector model")?;
        let mut head_model = core
            .compile_model(&head_network, DeviceType::CPU)
            .context("compiling the head-pose model for CPU")?;
        let mut gaze_model = core
            .compile_model(&gaze_network, DeviceType::CPU)
            .context("compiling the gaze-vector model for CPU")?;
        let head_request = head_model
            .create_infer_request()
            .context("creating the head-pose inference request")?;
        let gaze_request = gaze_model
            .create_infer_request()
            .context("creating the gaze-vector inference request")?;

        Ok(Self {
            head_request,
            gaze_request,
            _head_model: head_model,
            _gaze_model: gaze_model,
            _core: core,
            head_input: image_tensor()?,
            left_eye_input: image_tensor()?,
            right_eye_input: image_tensor()?,
            head_angles_input: tensor(&[1, 3])?,
            _runtime_dependencies: runtime_dependencies,
        })
    }

    /// Infer a calibrated feature from an RGB frame and its MediaPipe face geometry.
    ///
    /// `x` and `y` in the returned feature are the roll-compensated gaze vector projected on the
    /// camera plane.  They are deliberately not screen pixels: calibration learns the individual
    /// camera/display relationship from these real gaze-vector measurements.
    pub fn estimate(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        landmarks: &LandmarkFrame,
    ) -> Result<EyeFeatures> {
        let started = std::time::Instant::now();
        let landmark_confidence = crop_landmark_confidence(landmarks)?;
        if rgb.len() < width as usize * height as usize * 3 {
            bail!("camera RGB frame is shorter than its declared resolution")
        }
        let face = face_crop(landmarks, width, height)?;
        let mut left = eye_crop(landmarks, width, height, LEFT_EYE_OUTER, LEFT_EYE_INNER)?;
        let mut right = eye_crop(landmarks, width, height, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)?;
        let (left_iris, left_opening) =
            eye_geometry(landmarks, width, height, 33, 133, 159, 145, 468..473)?;
        let (right_iris, right_opening) =
            eye_geometry(landmarks, width, height, 362, 263, 386, 374, 473..478)?;
        let blink = left_opening < 0.075 || right_opening < 0.075;
        if blink {
            bail!("blink or occluded eye; gaze unavailable")
        }
        let image_score = eye_image_quality(rgb, width, height, left)?
            .min(eye_image_quality(rgb, width, height, right)?);
        if image_score < 0.35 {
            bail!("eye crops are too dark, overexposed, or lack contrast")
        }

        fill_image_tensor(&mut self.head_input, rgb, width, height, face)?;
        self.head_request
            .set_tensor("data", &self.head_input)
            .context("setting the head-pose input")?;
        self.head_request
            .infer()
            .context("running head-pose CPU inference")?;
        let angles = [
            scalar_output(&self.head_request, "angle_y_fc")?,
            scalar_output(&self.head_request, "angle_p_fc")?,
            scalar_output(&self.head_request, "angle_r_fc")?,
        ];
        if angles.iter().any(|angle| !angle.is_finite())
            || angles[0].abs() > 45.0
            || angles[1].abs() > 35.0
            || angles[2].abs() > 35.0
        {
            bail!("head-pose model produced an implausible angle")
        }

        // Same roll alignment as Intel's reference: rotate both crops by head
        // roll, pass zero roll to the network, then rotate its output back.
        left.rotation_radians = angles[2].to_radians();
        right.rotation_radians = left.rotation_radians;
        validate_crop(left, width, height)?;
        validate_crop(right, width, height)?;
        fill_image_tensor(&mut self.left_eye_input, rgb, width, height, left)?;
        fill_image_tensor(&mut self.right_eye_input, rgb, width, height, right)?;
        fill_tensor(&mut self.head_angles_input, &[angles[0], angles[1], 0.0])?;
        self.gaze_request
            .set_tensor("left_eye_image", &self.left_eye_input)
            .context("setting the left eye input")?;
        self.gaze_request
            .set_tensor("right_eye_image", &self.right_eye_input)
            .context("setting the right eye input")?;
        self.gaze_request
            .set_tensor("head_pose_angles", &self.head_angles_input)
            .context("setting the head-pose-angle input")?;
        self.gaze_request
            .infer()
            .context("running gaze-vector CPU inference")?;
        let vector_tensor = self
            .gaze_request
            .get_tensor("gaze_vector")
            .context("reading the gaze-vector output")?;
        let vector = vector_tensor
            .get_data::<f32>()
            .context("interpreting the gaze-vector output")?;
        if vector.len() < 3 || vector[..3].iter().any(|value| !value.is_finite()) {
            bail!("gaze-vector model returned invalid output")
        }

        // Gaze-estimation-adas-0002's vector is expressed in the camera reference frame. Undo
        // head roll before projection so a small head tilt does not look like a screen movement.
        let direction = normalize_gaze_vector([vector[0], vector[1], vector[2]], angles[2])
            .ok_or_else(|| anyhow!("invalid or backward-facing gaze direction"))?;
        // This is an image/geometry quality score, never a vector-norm probability.
        let confidence = image_score.min(landmark_confidence.unwrap_or(1.0));
        Ok(EyeFeatures {
            x: direction[0] / direction[2],
            y: direction[1] / direction[2],
            gaze_direction: direction,
            head_pose: angles.map(f64::from),
            left_iris: Some(left_iris),
            right_iris: Some(right_iris),
            face_center: Point::new(
                f64::from(face.center_x / width as f32),
                f64::from(face.center_y / height as f32),
            ),
            face_scale: f64::from(face.side / width.max(height) as f32),
            confidence,
            quality: Quality {
                landmark_confidence,
                image_score,
                eye_opening: [left_opening, right_opening],
                crops_valid: true,
            },
            // This binocular OpenVINO network needs two eye crops. A missing eye is rejected
            // instead of inventing a gaze coordinate from landmark position.
            one_eye_fallback: false,
            blink,
            timestamp_ms: landmarks.timestamp_ms,
            inference_latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

fn image_tensor() -> Result<Tensor> {
    tensor(&[1, 3, IMAGE_SIDE as i64, IMAGE_SIDE as i64])
}

fn tensor(shape: &[i64]) -> Result<Tensor> {
    let shape = Shape::new(shape).context("constructing an OpenVINO tensor shape")?;
    Tensor::new(ElementType::F32, &shape).context("allocating an OpenVINO tensor")
}

fn fill_tensor(tensor: &mut Tensor, data: &[f32]) -> Result<()> {
    let target = tensor
        .get_data_mut::<f32>()
        .context("getting writable OpenVINO tensor data")?;
    if target.len() != data.len() {
        bail!("OpenVINO tensor shape did not match its input data")
    }
    target.copy_from_slice(data);
    Ok(())
}

fn scalar_output(request: &InferRequest, name: &str) -> Result<f32> {
    let tensor = request
        .get_tensor(name)
        .with_context(|| format!("reading {name} output"))?;
    tensor
        .get_data::<f32>()
        .context("interpreting a head-pose output")?
        .first()
        .copied()
        .ok_or_else(|| anyhow!("{name} output was empty"))
}

#[derive(Clone, Copy)]
struct Crop {
    center_x: f32,
    center_y: f32,
    side: f32,
    rotation_radians: f32,
}

fn face_crop(frame: &LandmarkFrame, width: u32, height: u32) -> Result<Crop> {
    if frame.landmarks.len() < LANDMARK_COUNT {
        bail!("face crop requires all MediaPipe landmarks")
    }
    let points = frame.landmarks.iter().take(468).filter(|point| {
        point.x.is_finite()
            && point.y.is_finite()
            && (0.0..=1.0).contains(&point.x)
            && (0.0..=1.0).contains(&point.y)
    });
    let mut left = 1.0_f32;
    let mut right = 0.0_f32;
    let mut top = 1.0_f32;
    let mut bottom = 0.0_f32;
    let mut count = 0_usize;
    for point in points {
        left = left.min(point.x);
        right = right.max(point.x);
        top = top.min(point.y);
        bottom = bottom.max(point.y);
        count += 1;
    }
    if count < 100 {
        bail!("MediaPipe face geometry was incomplete")
    }
    let face_width = (right - left) * width as f32;
    let face_height = (bottom - top) * height as f32;
    let side = face_width.max(face_height) * 1.30;
    if side < 32.0 {
        bail!("face is too small for reliable gaze estimation")
    }
    Ok(Crop {
        center_x: ((left + right) * 0.5) * width as f32,
        center_y: ((top + bottom) * 0.5) * height as f32,
        side,
        rotation_radians: 0.0,
    })
}

fn eye_crop(
    frame: &LandmarkFrame,
    width: u32,
    height: u32,
    first: usize,
    second: usize,
) -> Result<Crop> {
    let first = landmark(frame, first)?;
    let second = landmark(frame, second)?;
    let first_x = first.x * width as f32;
    let first_y = first.y * height as f32;
    let second_x = second.x * width as f32;
    let second_y = second.y * height as f32;
    let dx = second_x - first_x;
    let dy = second_y - first_y;
    let distance = dx.hypot(dy);
    if distance < 8.0 {
        bail!("eye is too small for reliable gaze estimation")
    }
    Ok(Crop {
        center_x: (first_x + second_x) * 0.5,
        center_y: (first_y + second_y) * 0.5,
        side: distance * 1.85,
        rotation_radians: 0.0,
    })
}

fn landmark(frame: &LandmarkFrame, index: usize) -> Result<Landmark> {
    let point = frame
        .landmarks
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("MediaPipe landmark {index} was missing"))?;
    if !point.x.is_finite()
        || !point.y.is_finite()
        || !(0.0..=1.0).contains(&point.x)
        || !(0.0..=1.0).contains(&point.y)
    {
        bail!("MediaPipe landmark {index} was outside the camera image")
    }
    Ok(point)
}

fn eye_geometry(
    frame: &LandmarkFrame,
    width: u32,
    height: u32,
    outer: usize,
    inner: usize,
    upper: usize,
    lower: usize,
    iris: std::ops::Range<usize>,
) -> Result<(Point, f32)> {
    let pixel = |i| -> Result<Point> {
        let p = landmark(frame, i)?;
        Ok(Point::new(
            f64::from(p.x) * f64::from(width),
            f64::from(p.y) * f64::from(height),
        ))
    };
    let a = pixel(outer)?;
    let b = pixel(inner)?;
    let eye_width = a.distance_to(b);
    if eye_width < 8.0 {
        bail!("eye is too small")
    }
    // Both eyes use image-left to image-right coordinates, with no mirrored eye.
    let (a, b) = if a.x <= b.x { (a, b) } else { (b, a) };
    let u = Point::new((b.x - a.x) / eye_width, (b.y - a.y) / eye_width);
    let points: Vec<_> = iris.map(pixel).collect::<Result<_>>()?;
    let center = Point::new(
        points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
        points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
    );
    let displacement = Point::new(center.x - a.x, center.y - a.y);
    let normalized = Point::new(
        (displacement.x * u.x + displacement.y * u.y) / eye_width,
        (-displacement.x * u.y + displacement.y * u.x) / eye_width,
    );
    Ok((
        normalized,
        (pixel(upper)?.distance_to(pixel(lower)?) / eye_width) as f32,
    ))
}

fn validate_crop(crop: Crop, width: u32, height: u32) -> Result<()> {
    let half =
        crop.side * 0.5 * (crop.rotation_radians.cos().abs() + crop.rotation_radians.sin().abs());
    if crop.center_x - half < 0.0
        || crop.center_y - half < 0.0
        || crop.center_x + half >= width as f32 - 1.0
        || crop.center_y + half >= height as f32 - 1.0
    {
        bail!("eye crop crosses the camera boundary")
    }
    Ok(())
}

fn eye_image_quality(rgb: &[u8], width: u32, height: u32, crop: Crop) -> Result<f32> {
    validate_crop(crop, width, height)?;
    let mut values = Vec::with_capacity(144);
    for y in 0..12 {
        for x in 0..12 {
            let px = crop.center_x + crop.side * ((x as f32 + 0.5) / 12.0 - 0.5);
            let py = crop.center_y + crop.side * ((y as f32 + 0.5) / 12.0 - 0.5);
            let [r, g, b] = bilinear_rgb(rgb, width, height, px, py);
            values.push(0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b));
        }
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let clipped =
        values.iter().filter(|v| **v < 5.0 || **v > 250.0).count() as f32 / values.len() as f32;
    Ok((variance.sqrt() / 12.0)
        .clamp(0.0, 1.0)
        .min((mean / 35.0).clamp(0.0, 1.0))
        .min(((255.0 - mean) / 20.0).clamp(0.0, 1.0))
        * (1.0 - clipped))
}

fn crop_landmark_confidence(frame: &LandmarkFrame) -> Result<Option<f32>> {
    let mut confidence = frame
        .face_confidence
        .is_finite()
        .then_some(frame.face_confidence.clamp(0.0, 1.0));
    for index in [
        LEFT_EYE_OUTER,
        LEFT_EYE_INNER,
        RIGHT_EYE_INNER,
        RIGHT_EYE_OUTER,
    ] {
        let value = landmark(frame, index)?.confidence;
        if value.is_finite() {
            confidence = Some(confidence.unwrap_or(1.0).min(value.clamp(0.0, 1.0)));
        }
    }
    if confidence.is_some_and(|v| v < 0.50) {
        bail!("MediaPipe eye geometry confidence was too low for gaze crops")
    }
    Ok(confidence)
}

fn fill_image_tensor(
    tensor: &mut Tensor,
    rgb: &[u8],
    width: u32,
    height: u32,
    crop: Crop,
) -> Result<()> {
    let target = tensor
        .get_data_mut::<f32>()
        .context("getting writable image tensor data")?;
    if target.len() != IMAGE_VALUES {
        bail!("OpenVINO image tensor has an unexpected shape")
    }
    let cosine = crop.rotation_radians.cos();
    let sine = crop.rotation_radians.sin();
    for y in 0..IMAGE_SIDE {
        let local_y = (y as f32 + 0.5) / IMAGE_SIDE as f32 - 0.5;
        for x in 0..IMAGE_SIDE {
            let local_x = (x as f32 + 0.5) / IMAGE_SIDE as f32 - 0.5;
            let source_x = crop.center_x + crop.side * (local_x * cosine - local_y * sine);
            let source_y = crop.center_y + crop.side * (local_x * sine + local_y * cosine);
            let [red, green, blue] = bilinear_rgb(rgb, width, height, source_x, source_y);
            let offset = y * IMAGE_SIDE + x;
            // Open Model Zoo image inputs use BGR planar float tensors with pixel values 0..255.
            target[offset] = f32::from(blue);
            target[IMAGE_SIDE * IMAGE_SIDE + offset] = f32::from(green);
            target[2 * IMAGE_SIDE * IMAGE_SIDE + offset] = f32::from(red);
        }
    }
    Ok(())
}

fn bilinear_rgb(rgb: &[u8], width: u32, height: u32, x: f32, y: f32) -> [u8; 3] {
    if x < 0.0 || y < 0.0 || x >= width as f32 - 1.0 || y >= height as f32 - 1.0 {
        return [0, 0, 0];
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let sample = |sample_x: u32, sample_y: u32, channel: usize| {
        rgb[((sample_y * width + sample_x) as usize) * 3 + channel] as f32
    };
    let mut output = [0_u8; 3];
    for (channel, value) in output.iter_mut().enumerate() {
        let top = sample(x0, y0, channel) * (1.0 - dx) + sample(x1, y0, channel) * dx;
        let bottom = sample(x0, y1, channel) * (1.0 - dx) + sample(x1, y1, channel) * dx;
        *value = (top * (1.0 - dy) + bottom * dy).round().clamp(0.0, 255.0) as u8;
    }
    output
}

fn extract_assets(directory: &Path, assets: &[EmbeddedAsset]) -> Result<()> {
    fs::create_dir_all(directory)
        .with_context(|| format!("creating managed runtime directory {}", directory.display()))?;
    for asset in assets {
        if sha256_hex(asset.bytes) != asset.sha256 {
            bail!("embedded {} did not match its pinned SHA-256", asset.name)
        }
        let destination = directory.join(asset.name);
        if destination.exists() {
            let existing = fs::read(&destination)
                .with_context(|| format!("reading managed asset {}", destination.display()))?;
            if sha256_hex(&existing) == asset.sha256 {
                continue;
            }
            bail!(
                "managed asset {} has an unexpected hash; remove only that file and restart EyeOS",
                destination.display()
            )
        }
        fs::write(&destination, asset.bytes)
            .with_context(|| format!("writing managed asset {}", destination.display()))?;
    }
    Ok(())
}

fn load_runtime_dependencies(
    runtime_directory: &Path,
) -> Result<Vec<libloading::os::windows::Library>> {
    use libloading::os::windows::Library;
    use windows_sys::Win32::System::LibraryLoader::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };
    let directory =
        fs::canonicalize(runtime_directory).context("resolving native runtime directory")?;
    let mut libraries = Vec::new();
    // Full-path loads retain dependencies without changing the host's DLL search directory.
    for name in [
        "tbb12.dll",
        "tbbmalloc.dll",
        "openvino.dll",
        "openvino_intel_cpu_plugin.dll",
        "openvino_ir_frontend.dll",
    ] {
        let library = unsafe {
            Library::load_with_flags(
                directory.join(name),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .with_context(|| format!("loading native dependency {name}"))?;
        libraries.push(library);
    }
    Ok(libraries)
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_crop_reads_rgb_without_channel_swapping() {
        let rgb = [
            10, 20, 30, 50, 60, 70, // top row
            90, 100, 110, 130, 140, 150, // bottom row
        ];
        assert_eq!(bilinear_rgb(&rgb, 2, 2, 0.0, 0.0), [10, 20, 30]);
        assert_eq!(bilinear_rgb(&rgb, 2, 2, 0.5, 0.5), [70, 80, 90]);
        assert_eq!(bilinear_rgb(&rgb, 2, 2, -1.0, 0.5), [0, 0, 0]);
    }

    #[test]
    fn embedded_assets_match_pinned_hashes() {
        for asset in OPENVINO_DLLS.iter().chain(MODEL_ASSETS) {
            assert_eq!(sha256_hex(asset.bytes), asset.sha256, "{}", asset.name);
        }
    }

    #[cfg(windows)]
    #[test]
    fn bundled_runtime_compiles_and_runs_both_cpu_models() {
        let directory = tempfile::tempdir().expect("temporary managed runtime directory");
        let mut estimator = GazeEstimator::load(directory.path())
            .expect("bundled OpenVINO CPU runtime and gaze models must initialise");
        let mut landmarks = vec![
            Landmark {
                x: 0.5,
                y: 0.5,
                z: 0.0,
                confidence: 1.0,
            };
            LANDMARK_COUNT
        ];
        landmarks[10] = Landmark {
            y: 0.20,
            ..landmarks[10]
        };
        landmarks[152] = Landmark {
            y: 0.80,
            ..landmarks[152]
        };
        landmarks[234] = Landmark {
            x: 0.25,
            ..landmarks[234]
        };
        landmarks[454] = Landmark {
            x: 0.75,
            ..landmarks[454]
        };
        landmarks[LEFT_EYE_OUTER] = Landmark {
            x: 0.36,
            y: 0.45,
            ..landmarks[LEFT_EYE_OUTER]
        };
        landmarks[LEFT_EYE_INNER] = Landmark {
            x: 0.46,
            y: 0.45,
            ..landmarks[LEFT_EYE_INNER]
        };
        landmarks[RIGHT_EYE_INNER] = Landmark {
            x: 0.54,
            y: 0.45,
            ..landmarks[RIGHT_EYE_INNER]
        };
        landmarks[RIGHT_EYE_OUTER] = Landmark {
            x: 0.64,
            y: 0.45,
            ..landmarks[RIGHT_EYE_OUTER]
        };
        for (upper, lower) in [(159, 145), (386, 374)] {
            landmarks[upper].y = 0.43;
            landmarks[lower].y = 0.47;
        }
        let frame = LandmarkFrame {
            landmarks,
            timestamp_ms: 0,
            face_confidence: 1.0,
        };
        // Real quality gates reject constant grey images. This fixture verifies
        // preprocessing/runtime plumbing, not gaze accuracy on an artificial face.
        let rgb: Vec<u8> = (0..640 * 480)
            .flat_map(|i| {
                let value = if ((i % 640) / 4 + (i / 640) / 4) % 2 == 0 {
                    70
                } else {
                    190
                };
                [value, value, value]
            })
            .collect();
        let feature = estimator
            .estimate(&rgb, 640, 480, &frame)
            .expect("both local models must run on a valid RGB frame");
        assert!(feature.x.is_finite() && feature.y.is_finite());
        assert!(feature.confidence > 0.0);
    }
}
