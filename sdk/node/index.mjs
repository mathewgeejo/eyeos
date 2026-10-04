import koffi from 'koffi';
import { fileURLToPath } from 'node:url';
import { threadId } from 'node:worker_threads';

export class TrackerError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
// Define once: koffi struct names are process-global.
const EyeBuffer = koffi.struct('EyeTrackerBufferV1', { data: 'void *', len: 'size_t' });

export class Tracker {
  constructor(config = {}, library = process.env.EYE_TRACKER_LIBRARY) {
    this.thread = threadId;
    this.handle = null;
    library ??= fileURLToPath(new URL('../native/eyeos.dll', import.meta.url));
    this.lib = koffi.load(library);
    this.fn = {};
    const declarations = {
      et_abi_version: 'uint32_t et_abi_version()',
      et_create: 'int32_t et_create(const uint8_t *, size_t, _Out_ uint64_t *)',
      et_destroy: 'int32_t et_destroy(uint64_t)',
      et_start_camera: 'int32_t et_start_camera(uint64_t, uint32_t)',
      et_stop_camera: 'int32_t et_stop_camera(uint64_t)',
      et_timestamp_ms: 'int32_t et_timestamp_ms(uint64_t, _Out_ uint64_t *)',
      et_start_calibration: 'int32_t et_start_calibration(uint64_t)',
      et_extend_calibration: 'int32_t et_extend_calibration(uint64_t)',
      et_cancel_calibration: 'int32_t et_cancel_calibration(uint64_t)',
      et_import_profile: 'int32_t et_import_profile(uint64_t, const uint8_t *, size_t)',
      et_process_rgb: 'int32_t et_process_rgb(uint64_t, const uint8_t *, size_t, uint32_t, uint32_t, size_t, uint64_t, _Out_ EyeTrackerBufferV1 *)',
      et_process_observation: 'int32_t et_process_observation(uint64_t, const uint8_t *, size_t, _Out_ EyeTrackerBufferV1 *)',
      et_poll: 'int32_t et_poll(uint64_t, _Out_ EyeTrackerBufferV1 *)',
      et_calibration_progress: 'int32_t et_calibration_progress(uint64_t, _Out_ EyeTrackerBufferV1 *)',
      et_export_profile: 'int32_t et_export_profile(uint64_t, _Out_ EyeTrackerBufferV1 *)',
      et_last_error: 'int32_t et_last_error(_Out_ EyeTrackerBufferV1 *)',
      et_buffer_free: 'int32_t et_buffer_free(_Inout_ EyeTrackerBufferV1 *)',
    };
    for (const [name, declaration] of Object.entries(declarations)) this.fn[name] = this.lib.func(declaration);
    if (this.fn.et_abi_version() !== 1) throw new TrackerError(4, 'Unsupported native SDK ABI');
    const data = Buffer.from(JSON.stringify(config));
    const handle = [0];
    this.check(this.fn.et_create(data, data.length, handle));
    this.handle = handle[0];
  }
  owningThread() {
    if (threadId !== this.thread) throw new TrackerError(2, 'Use the tracker on its creating thread');
    if (this.handle === null) throw new TrackerError(2, 'Tracker is closed');
  }
  read(buffer) {
    try {
      const bytes = koffi.decode(buffer.data, 'uint8_t', Number(buffer.len));
      return JSON.parse(Buffer.from(bytes).toString('utf8'));
    } finally { this.fn.et_buffer_free(buffer); }
  }
  check(status) {
    if (status) {
      const buffer = { data: null, len: 0 };
      if (this.fn.et_last_error(buffer) === 0) {
        const error = this.read(buffer);
        throw new TrackerError(error.code, error.message);
      }
      throw new TrackerError(status, 'Native tracker operation failed');
    }
  }
  call(name, ...args) { this.owningThread(); this.check(this.fn[name](this.handle, ...args)); }
  result(name, ...args) {
    this.owningThread();
    const buffer = { data: null, len: 0 };
    this.check(this.fn[name](this.handle, ...args, buffer));
    return this.read(buffer);
  }
  startCamera(index = 0) { this.call('et_start_camera', index); }
  stopCamera() { this.call('et_stop_camera'); }
  timestampMs() { const value = [0]; this.call('et_timestamp_ms', value); return Number(value[0]); }
  poll() { return this.result('et_poll'); }
  startCalibration() { this.call('et_start_calibration'); }
  extendCalibration() { this.call('et_extend_calibration'); }
  cancelCalibration() { this.call('et_cancel_calibration'); }
  calibrationProgress() { return this.result('et_calibration_progress'); }
  processRgb(rgb, width, height, timestampMs, stride = width * 3) {
    const bytes = Buffer.from(rgb);
    return this.result('et_process_rgb', bytes, bytes.length, width, height, stride, timestampMs);
  }
  processObservation(observation) {
    const data = Buffer.from(JSON.stringify(observation));
    return this.result('et_process_observation', data, data.length);
  }
  importProfile(profile) {
    const data = Buffer.from(JSON.stringify(profile));
    this.call('et_import_profile', data, data.length);
  }
  exportProfile() { return this.result('et_export_profile'); }
  close() { if (this.handle !== null) { this.call('et_destroy'); this.handle = null; } }
}
