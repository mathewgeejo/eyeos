const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('eyeTracker', {
  calibrate: () => ipcRenderer.invoke('calibrate'),
  extend: () => ipcRenderer.invoke('extend'),
  quit: () => ipcRenderer.invoke('quit'),
  subscribe: callback => ipcRenderer.on('tracker-events', (_, event) => callback(event)),
});
