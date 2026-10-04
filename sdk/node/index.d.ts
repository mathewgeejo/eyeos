export interface Point { x: number; y: number }
export interface TrackerConfig {
  screen_size?: Point; screen_width_mm?: number; screen_height_mm?: number;
  viewing_distance_mm?: number; camera_id?: string; display_id?: string;
  model_id?: string; minimum_quality?: number; maximum_frame_age_ms?: number;
  median_target_deg?: number; p95_target_deg?: number; runtime_root?: string;
}
export interface Quality {
  landmark_confidence: number | null; image_score: number;
  eye_opening: [number, number]; crops_valid: boolean;
}
export interface Observation {
  x: number; y: number; gaze_direction?: [number, number, number];
  head_pose?: [number, number, number]; left_iris?: Point | null; right_iris?: Point | null;
  face_center?: Point; face_scale?: number; confidence: number; quality?: Quality;
  one_eye_fallback?: boolean; blink?: boolean; timestamp_ms: number; inference_latency_ms?: number;
}
export interface GazeEstimate {
  timestamp_ms: number; raw: Point | null; filtered: Point | null; quality: Quality;
  quality_score: number; estimated_error_px: number | null;
  state: 'Tracking' | 'Uncalibrated' | 'CalibrationRequired' | 'TrackingLost' | 'StaleFrame';
  precision_validated: boolean; retrieval_used: boolean; latency_ms: number;
}
export interface AccuracyReport {
  median_error_px: number; p95_error_px: number;
  median_error_deg: number | null; p95_error_deg: number | null;
  valid_sample_coverage: number; jitter_px: number;
  median_latency_ms: number; p95_latency_ms: number; samples: number; precision_passed: boolean;
  regions: { region: string; median_px: number; p95_px: number; samples: number }[];
}
export interface CalibrationProgress {
  target: Point | null; phase: string; instruction: string;
  completed: number; total: number; stable_samples: number; suggested_targets: Point[];
}
export interface Profile {
  version: number; config: TrackerConfig; validation_passed: boolean;
  accuracy: AccuracyReport | null; [key: string]: unknown;
}
export interface SdkEvent {
  estimate: GazeEstimate | null; observation: Observation | null;
  calibration: { Completed: { profile: Profile; report: AccuracyReport } } | { Rejected: string } | null;
  progress: CalibrationProgress | null; status: unknown;
}
export class TrackerError extends Error { code: number; constructor(code: number, message: string) }
export class Tracker {
  constructor(config?: TrackerConfig, library?: string);
  startCamera(index?: number): void; stopCamera(): void; timestampMs(): number;
  poll(): SdkEvent[]; startCalibration(): void; extendCalibration(): void; cancelCalibration(): void;
  calibrationProgress(): CalibrationProgress | null;
  processRgb(rgb: Uint8Array, width: number, height: number, timestampMs: number, stride?: number): SdkEvent;
  processObservation(observation: Observation): SdkEvent;
  importProfile(profile: Profile): void; exportProfile(): Profile; close(): void;
}
