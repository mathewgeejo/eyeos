import { Tracker } from './index.mjs';
const tracker = new Tracker();
tracker.startCamera();
const timer = setInterval(() => {
  for (const event of tracker.poll()) console.log(JSON.stringify(event));
}, 16);
function close() { clearInterval(timer); tracker.close(); process.exit(0); }
process.on('SIGINT', close);
setTimeout(close, 5000);
