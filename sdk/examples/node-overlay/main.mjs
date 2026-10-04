import { app, BrowserWindow, screen, ipcMain } from 'electron';
import { Tracker } from '../../node/index.mjs';
import { fileURLToPath } from 'node:url';
let tracker, window, timer;
await app.whenReady();
const display = screen.getPrimaryDisplay();
const scale = display.scaleFactor;
const width = Math.round(display.size.width * scale), height = Math.round(display.size.height * scale);
const measurement = key => process.env[key] ? Number(process.env[key]) : undefined;
tracker = new Tracker({
  screen_size: { x: width, y: height },
  screen_width_mm: measurement('SCREEN_WIDTH_MM'), screen_height_mm: measurement('SCREEN_HEIGHT_MM'),
  viewing_distance_mm: measurement('VIEWING_DISTANCE_MM'),
});
window = new BrowserWindow({ fullscreen: true, backgroundColor: '#101820',
  webPreferences: { preload: fileURLToPath(new URL('./preload.cjs', import.meta.url)),
    contextIsolation: true, nodeIntegration: false } });
ipcMain.handle('calibrate', () => { tracker.startCalibration(); });
ipcMain.handle('extend', () => { tracker.extendCalibration(); });
ipcMain.handle('quit', () => { app.quit(); });
await window.loadFile(fileURLToPath(new URL('./index.html', import.meta.url)));
tracker.startCamera();
timer = setInterval(() => {
  if (!window.isDestroyed()) window.webContents.send('tracker-events', { scale, events: tracker.poll(), progress: tracker.calibrationProgress() });
}, 16);
app.on('before-quit', () => { clearInterval(timer); tracker?.close(); });
