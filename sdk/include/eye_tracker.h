#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1; all calls on a handle must run on its creating thread.
   Input data is borrowed for the call. Outputs are UTF-8 JSON, not NUL terminated.
   Initialize EyeBuffer to {0}; free each successful output exactly once.
   Never copy/free its allocation with the host allocator. No function injects input. */
typedef uint64_t EyeTrackerHandle;
typedef struct EyeBuffer { uint8_t *data; size_t len; } EyeBuffer;
enum EyeTrackerError { ET_OK=0, ET_ARGUMENT=1, ET_HANDLE=2, ET_STATE=3, ET_RUNTIME=4, ET_PANIC=5 };
uint32_t et_abi_version(void);
int32_t et_create(const uint8_t *config_json, size_t len, EyeTrackerHandle *out);
int32_t et_destroy(EyeTrackerHandle handle);
int32_t et_start_camera(EyeTrackerHandle handle, uint32_t index);
int32_t et_stop_camera(EyeTrackerHandle handle);
int32_t et_timestamp_ms(EyeTrackerHandle handle, uint64_t *out);
int32_t et_start_calibration(EyeTrackerHandle handle);
int32_t et_extend_calibration(EyeTrackerHandle handle);
int32_t et_cancel_calibration(EyeTrackerHandle handle);
int32_t et_calibration_progress(EyeTrackerHandle handle, EyeBuffer *out);
int32_t et_poll(EyeTrackerHandle handle, EyeBuffer *out);
int32_t et_process_rgb(EyeTrackerHandle handle, const uint8_t *rgb, size_t len,
    uint32_t width, uint32_t height, size_t stride, uint64_t timestamp_ms, EyeBuffer *out);
int32_t et_process_observation(EyeTrackerHandle handle, const uint8_t *json, size_t len, EyeBuffer *out);
int32_t et_import_profile(EyeTrackerHandle handle, const uint8_t *json, size_t len);
int32_t et_export_profile(EyeTrackerHandle handle, EyeBuffer *out);
/* Reads thread-local {code,message}; call immediately after an unsuccessful operation. */
int32_t et_last_error(EyeBuffer *out);
int32_t et_buffer_free(EyeBuffer *buffer);
#ifdef __cplusplus
}
#endif
#endif
