"""python example_overlay.py --width-mm 530 --height-mm 300 --distance-mm 600
Uses engine-owned camera capture; draws a target/cursor without moving the OS mouse.
"""
import argparse
import ctypes
import json
import os
import tkinter as tk
from pathlib import Path
from eye_tracker import Tracker, TrackerError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width-mm", type=float)
    parser.add_argument("--height-mm", type=float)
    parser.add_argument("--distance-mm", type=float)
    parser.add_argument("--save-profile", type=Path)
    args = parser.parse_args()
    if os.name == "nt":
        # Set BEFORE creating Tk so physical calibration pixels match the screen.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.title("Local eye tracker: calibration and gaze preview")
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    canvas = tk.Canvas(root, width=width, height=height, bg="#101820", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    tracker = Tracker({"screen_size": {"x": width, "y": height},
                       "camera_id": f"webcam:{args.camera}",
                       "screen_width_mm": args.width_mm, "screen_height_mm": args.height_mm,
                       "viewing_distance_mm": args.distance_mm}, args.library)
    status = canvas.create_text(width / 2, 45, fill="white", font=("Arial", 18),
                                text="Press C to calibrate; E for extra calibration; Escape to exit")
    target = canvas.create_oval(0, 0, 0, 0, fill="#44e6bc", outline="white", width=3, state="hidden")
    gaze = canvas.create_oval(0, 0, 0, 0, outline="#ffd36a", width=3, state="hidden")

    def start(_=None):
        tracker.start_calibration()

    def extend(_=None):
        try:
            tracker.extend_calibration()
        except TrackerError as error:
            canvas.itemconfigure(status, text=str(error))

    def close(_=None):
        tracker.close()
        root.destroy()

    def tick():
        try:
            for event in tracker.poll():
                estimate = event.get("estimate")
                if estimate and estimate.get("filtered"):
                    point = estimate["filtered"]
                    x, y = point["x"], point["y"]
                    canvas.coords(gaze, x - 12, y - 12, x + 12, y + 12)
                    canvas.itemconfigure(gaze, state="normal")
                elif estimate:
                    canvas.itemconfigure(gaze, state="hidden")
                outcome = event.get("calibration")
                if outcome and "Completed" in outcome:
                    report = outcome["Completed"]["report"]
                    text = f"Median {report['median_error_px']:.1f}px; p95 {report['p95_error_px']:.1f}px. "
                    text += "Precision passed." if report["precision_passed"] else "Precision not validated. Press E for extra setup."
                    canvas.itemconfigure(status, text=text)
                    if args.save_profile:
                        args.save_profile.write_text(json.dumps(tracker.export_profile(), indent=2), encoding="utf-8")
                elif outcome and "Rejected" in outcome:
                    canvas.itemconfigure(status, text=outcome["Rejected"])
            progress = tracker.calibration_progress()
            if progress and progress["target"]:
                x, y = progress["target"]["x"], progress["target"]["y"]
                canvas.coords(target, x - 22, y - 22, x + 22, y + 22)
                canvas.itemconfigure(target, state="normal")
                canvas.itemconfigure(status, text=f"{progress['instruction']} ({progress['completed'] + 1}/{progress['total']})")
            else:
                canvas.itemconfigure(target, state="hidden")
        except TrackerError as error:
            canvas.itemconfigure(status, text=str(error))
        root.after(16, tick)

    root.bind("c", start)
    root.bind("e", extend)
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    tracker.start_camera(args.camera)
    root.after(16, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
