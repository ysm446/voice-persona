const { app, BrowserWindow, ipcMain, Menu } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");
const http = require("http");

let mainWindow = null;
let pythonProcess = null;
let serverPort = 7860;

function findAvailablePort(startPort) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(startPort, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", () => {
      findAvailablePort(startPort + 1).then(resolve).catch(reject);
    });
  });
}

function waitForServer(port, maxAttempts = 60) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const check = () => {
      const req = http.get(`http://127.0.0.1:${port}/api/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          retry();
        }
        res.resume();
      });
      req.on("error", retry);
      req.setTimeout(1000, () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      attempts++;
      if (attempts >= maxAttempts) {
        reject(new Error("Python server failed to start"));
      } else {
        setTimeout(check, 1000);
      }
    };
    check();
  });
}

function startPythonServer(port) {
  const serverScript = path.join(__dirname, "..", "server.py");
  pythonProcess = spawn("python", [serverScript], {
    env: { ...process.env, APP_PORT: String(port) },
    cwd: path.join(__dirname, ".."),
  });

  pythonProcess.stdout.on("data", (data) => {
    process.stdout.write("[python] " + data);
  });
  pythonProcess.stderr.on("data", (data) => {
    process.stderr.write("[python] " + data);
  });
  pythonProcess.on("close", (code) => {
    console.log(`[python] process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 820,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
    },
    title: "Voice Persona",
    backgroundColor: "#1e1e1e",
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"), {
    query: { port: String(serverPort) },
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

ipcMain.handle("get-api-base", () => `http://127.0.0.1:${serverPort}`);

app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  try {
    serverPort = await findAvailablePort(7860);
    console.log(`[electron] Using port ${serverPort}`);
    startPythonServer(serverPort);
    console.log("[electron] Waiting for Python server...");
    await waitForServer(serverPort);
    console.log("[electron] Server ready, opening window");
    createWindow();
  } catch (err) {
    console.error("[electron] Startup failed:", err);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
  app.quit();
});

app.on("before-quit", () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
});
