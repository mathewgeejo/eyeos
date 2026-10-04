import test from 'node:test';
import assert from 'node:assert/strict';
import { Tracker, TrackerError } from './index.mjs';
import { readFileSync } from 'node:fs';

const library = process.env.EYE_TRACKER_LIBRARY;
test('shared profile produces the same Rust/C/Python/Node coordinates', { skip: !library }, () => {
  const profile = JSON.parse(readFileSync(new URL('../tests/profile-v2.json', import.meta.url), 'utf8'));
  const tracker = new Tracker({}, library);
  try {
    tracker.importProfile(profile);
    const event = tracker.processObservation({ x: 0.25, y: 0.75, confidence: 1,
      gaze_direction: [0, 0, 1], timestamp_ms: tracker.timestampMs() });
    assert.deepEqual(event.estimate.raw, { x: 480, y: 810 });
    assert.deepEqual(event.estimate.filtered, event.estimate.raw);
    assert.equal(event.estimate.precision_validated, false);
  } finally { tracker.close(); }
});
test('native ABI lifecycle, errors, and observation semantics', { skip: !library }, () => {
  const tracker = new Tracker({}, library);
  try {
    assert.deepEqual(tracker.poll(), []);
    assert.throws(() => tracker.exportProfile(), error => error instanceof TrackerError && error.code === 3);
    assert.throws(() => tracker.importProfile({ version: 1 }), TrackerError);
    const timestamp = tracker.timestampMs();
    const event = tracker.processObservation({ x: 0.5, y: 0.5, confidence: 1, gaze_direction: [0, 0, 1], timestamp_ms: timestamp });
    assert.equal(event.estimate.state, 'Uncalibrated');
    assert.equal(event.estimate.precision_validated, false);
    assert.equal(event.estimate.raw, null);
    tracker.startCalibration();
    const progress = tracker.calibrationProgress();
    assert.deepEqual(progress.target, { x: 153.6, y: 86.4 });
    assert.equal(progress.total, 12);
    assert.throws(() => tracker.extendCalibration(), TrackerError);
    tracker.cancelCalibration();
    assert.equal(tracker.calibrationProgress(), null);
  } finally { tracker.close(); }
  assert.throws(() => tracker.poll(), error => error.code === 2);
  tracker.close();
});
