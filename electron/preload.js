const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getApiBase: () => ipcRenderer.invoke("get-api-base"),
});
