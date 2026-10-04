import os
import unittest
import json
from pathlib import Path
from eye_tracker import Tracker, TrackerError


@unittest.skipUnless(os.environ.get("EYE_TRACKER_LIBRARY"), "Set EYE_TRACKER_LIBRARY to test the native ABI")
class NativeSdkTests(unittest.TestCase):
    def test_shared_profile_coordinates(self):
        profile = json.loads((Path(__file__).resolve().parents[1] / "tests" / "profile-v2.json").read_text(encoding="utf-8"))
        with Tracker() as tracker:
            tracker.import_profile(profile)
            event = tracker.process_observation({"x": 0.25, "y": 0.75, "confidence": 1,
                "gaze_direction": [0, 0, -1], "timestamp_ms": tracker.timestamp_ms()})
            self.assertEqual(event["estimate"]["raw"], {"x": 480.0, "y": 810.0})
            self.assertEqual(event["estimate"]["filtered"], event["estimate"]["raw"])
            self.assertFalse(event["estimate"]["precision_validated"])

    def test_lifecycle_errors_and_observation_semantics(self):
        with Tracker() as tracker:
            self.assertEqual(tracker.poll(), [])
            with self.assertRaises(TrackerError) as error:
                tracker.export_profile()
            self.assertEqual(error.exception.code, 3)
            with self.assertRaises(TrackerError):
                tracker.import_profile({"version": 1})
            event = tracker.process_observation({"x": 0.5, "y": 0.5, "confidence": 1,
                                                 "gaze_direction": [0, 0, -1],
                                                 "timestamp_ms": tracker.timestamp_ms()})
            self.assertEqual(event["estimate"]["state"], "Uncalibrated")
            self.assertFalse(event["estimate"]["precision_validated"])
            self.assertIsNone(event["estimate"]["raw"])
            tracker.start_calibration()
            progress = tracker.calibration_progress()
            self.assertEqual(progress["target"], {"x": 153.6, "y": 86.4})
            self.assertEqual(progress["total"], 12)
            with self.assertRaises(TrackerError):
                tracker.extend_calibration()
            tracker.cancel_calibration()
            self.assertIsNone(tracker.calibration_progress())
        with self.assertRaises(TrackerError) as error:
            tracker.poll()
        self.assertEqual(error.exception.code, 2)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
