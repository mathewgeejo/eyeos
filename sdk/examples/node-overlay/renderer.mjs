const canvas = document.querySelector('canvas'), context = canvas.getContext('2d'), status = document.querySelector('#status');
let gaze;
for (const action of ['calibrate', 'extend', 'quit']) {
  document.querySelector(`#${action}`).onclick = () => window.eyeTracker[action]().catch(error => { status.textContent = error.message; });
}
window.eyeTracker.subscribe(({ scale, events, progress }) => {
  canvas.width = innerWidth; canvas.height = innerHeight;
  for (const event of events) {
    if (event.estimate) gaze = event.estimate.filtered;
    if (event.calibration?.Completed) {
      const report = event.calibration.Completed.report;
      status.textContent = `Median ${report.median_error_px.toFixed(1)}px, p95 ${report.p95_error_px.toFixed(1)}px. ${report.precision_passed ? 'Precision passed.' : 'Precision not validated; try extra calibration.'}`;
    }
    if (event.calibration?.Rejected) status.textContent = event.calibration.Rejected;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  function circle(point, radius, color) {
    context.beginPath(); context.arc(point.x / scale, point.y / scale, radius, 0, 2 * Math.PI);
    context.strokeStyle = color; context.lineWidth = 3; context.stroke();
  }
  if (gaze) circle(gaze, 12, '#ffd36a');
  if (progress?.target) {
    circle(progress.target, 22, '#44e6bc');
    status.textContent = `${progress.instruction} (${progress.completed + 1}/${progress.total})`;
  }
});
