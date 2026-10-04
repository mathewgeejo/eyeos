"""Standard-library wrapper for ABI v1. Keep each Tracker on its owning thread."""
import ctypes as C
import json
import os
import threading
from pathlib import Path


class TrackerError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _Buffer(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("len", C.c_size_t)]


class Tracker:
    def __init__(self, config=None, library=None):
        self._thread = threading.get_ident()
        self._handle = C.c_uint64()
        path = library or os.environ.get("EYE_TRACKER_LIBRARY")
        if not path:
            suffix = "eyeos.dll" if os.name == "nt" else "libeyeos.so"
            path = Path(__file__).resolve().parent.parent / "native" / suffix
        # Rust extern C uses cdecl, including on 32-bit Windows.
        self._lib = C.CDLL(str(path))
        self._bind()
        if self._lib.et_abi_version() != 1:
            raise TrackerError(4, "Unsupported native SDK ABI")
        data = self._json(config or {})
        self._check(self._lib.et_create(data, len(data), C.byref(self._handle)))

    def _bind(self):
        lib = self._lib
        h, p, n, out = C.c_uint64, C.c_void_p, C.c_size_t, C.POINTER(_Buffer)
        signatures = {
            "et_create": [p, n, C.POINTER(h)], "et_destroy": [h],
            "et_start_camera": [h, C.c_uint32], "et_stop_camera": [h],
            "et_timestamp_ms": [h, C.POINTER(h)],
            "et_start_calibration": [h], "et_extend_calibration": [h], "et_cancel_calibration": [h],
            "et_calibration_progress": [h, out], "et_poll": [h, out],
            "et_process_rgb": [h, p, n, C.c_uint32, C.c_uint32, n, h, out],
            "et_process_observation": [h, p, n, out],
            "et_import_profile": [h, p, n], "et_export_profile": [h, out],
            "et_last_error": [out], "et_buffer_free": [out],
        }
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, C.c_int32
        lib.et_abi_version.argtypes, lib.et_abi_version.restype = [], C.c_uint32

    @staticmethod
    def _json(value):
        return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def _owning_thread(self):
        if threading.get_ident() != self._thread:
            raise TrackerError(2, "Use the tracker on its creating thread")
        if not self._handle.value:
            raise TrackerError(2, "Tracker is closed")

    def _read(self, buffer):
        try:
            return json.loads(C.string_at(buffer.data, buffer.len).decode("utf-8"))
        finally:
            self._lib.et_buffer_free(C.byref(buffer))

    def _check(self, status):
        if status:
            buffer = _Buffer()
            if self._lib.et_last_error(C.byref(buffer)) == 0:
                error = self._read(buffer)
                raise TrackerError(error["code"], error["message"])
            raise TrackerError(status, "Native tracker operation failed")

    def _call(self, name, *args):
        self._owning_thread()
        self._check(getattr(self._lib, name)(self._handle, *args))

    def _result(self, name, *args):
        self._owning_thread()
        buffer = _Buffer()
        self._check(getattr(self._lib, name)(self._handle, *args, C.byref(buffer)))
        return self._read(buffer)

    def start_camera(self, index=0):
        self._call("et_start_camera", index)

    def stop_camera(self):
        self._call("et_stop_camera")

    def timestamp_ms(self):
        timestamp = C.c_uint64()
        self._call("et_timestamp_ms", C.byref(timestamp))
        return timestamp.value

    def poll(self):
        return self._result("et_poll")

    def start_calibration(self):
        self._call("et_start_calibration")

    def extend_calibration(self):
        self._call("et_extend_calibration")

    def cancel_calibration(self):
        self._call("et_cancel_calibration")

    def calibration_progress(self):
        return self._result("et_calibration_progress")

    def process_rgb(self, rgb, width, height, timestamp_ms, stride=None):
        # Copy to a contiguous borrowed buffer whose lifetime covers inference.
        data = bytes(rgb)
        return self._result("et_process_rgb", data, len(data), width, height,
                            stride or width * 3, timestamp_ms)

    def process_observation(self, observation):
        data = self._json(observation)
        return self._result("et_process_observation", data, len(data))

    def import_profile(self, profile):
        data = self._json(profile)
        self._call("et_import_profile", data, len(data))

    def export_profile(self):
        return self._result("et_export_profile")

    def close(self):
        if self._handle.value:
            self._call("et_destroy")
            self._handle.value = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
