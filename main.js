const {
  app,
  BrowserWindow,
  Menu,
  MenuItem,
  ipcMain,
  dialog,
  shell,
  systemPreferences,
  nativeTheme,
} = require('electron');
const electronLocalShortcut = require('electron-localshortcut');

app.setName('CodeCompanion.AI');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const ElectronStore = require('electron-store');
const pty = require('node-pty');
const { debounce } = require('lodash');
const Sentry = require('@sentry/electron');
const { initialize } = require('@aptabase/electron/main'); // for DAU tracking
const WindowManager = require('./app/window/WindowManager');

ElectronStore.initRenderer();
const localStorage = new ElectronStore();

let win;
let isUpdateInProgress = false;
let terminal;
let windowManager;

if (process.env.NODE_ENV === 'development' && !app.isPackaged) {
  setTimeout(() => {
    win.webContents.openDevTools();
  }, 3000);
} else {
  Sentry.init({
    dsn: 'https://87985c08c00b4f0c83989b182e9fbe95@o4505507137847296.ingest.sentry.io/4505507139485696',
  });
}
initialize('A-US-5249376059');

// Setup local shortcuts
function setupLocalShortcuts() {
  const shortcuts = [
    { key: 'CmdOrCtrl+M', action: () => windowManager.minimize() },
    { key: 'CmdOrCtrl+Alt+F', action: () => windowManager.maximize() },
    { key: 'F11', action: () => windowManager.toggleFullScreen() },
    { key: 'CmdOrCtrl+Shift+Left', action: () => windowManager.resizeWindow('narrower') },
    { key: 'CmdOrCtrl+Shift+Right', action: () => windowManager.resizeWindow('wider') },
    { key: 'CmdOrCtrl+Shift+Up', action: () => windowManager.resizeWindow('taller') },
    { key: 'CmdOrCtrl+Shift+Down', action: () => windowManager.resizeWindow('shorter') },
    { key: 'CmdOrCtrl+W', action: () => windowManager.close() },
    { key: 'Alt+Space', action: () => win.show() },
  ];

  // Add Mac-specific shortcuts
  if (process.platform === 'darwin') {
    shortcuts.push({ key: 'Cmd+Ctrl+F', action: () => windowManager.toggleFullScreen() });
  }

  // Add Windows-specific shortcuts
  if (process.platform === 'win32') {
    shortcuts.push({ key: 'Alt+Space', action: () => windowManager.showSystemMenu() });
  }

  function registerShortcuts() {
    shortcuts.forEach(({ key, action }) => {
      electronLocalShortcut.register(win, key, action);
    });
  }

  function unregisterShortcuts() {
    electronLocalShortcut.unregisterAll(win);
  }

  win.on('focus', registerShortcuts);
  win.on('blur', unregisterShortcuts);

  // Initial registration
  registerShortcuts();
}

function createWindow() {
  const { screen } = require('electron');
  let { width, height, x, y } = localStorage.get('windowBounds') || {};
  if (!width || !height) {
    const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().workAreaSize;
    screenWidth > 1400 ? (width = 1400) : (width = Math.floor(screenWidth * 0.8));
    screenHeight > 1080 ? (height = 1080) : (height = Math.floor(screenHeight * 0.8));
    localStorage.set('windowBounds', { width, height });
  }
  win = new BrowserWindow({
    show: false,
    width,
    height,
    x,
    y,
    frame: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      webviewTag: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  win.loadFile('index.html');

  // Initialize WindowManager
  windowManager = new WindowManager(win);

  // Setup local shortcuts
  setupLocalShortcuts();

  win.once('ready-to-show', () => {
    win.show();
  });

  const menuTemplate = [
    {
      label: 'CodeCompanion.AI',
      submenu: [
        {
          label: 'Open Project',
          accelerator: 'CmdOrCtrl+O',
          click() {
            win.webContents.executeJavaScript('viewController.selectDirectory();');
          },
        },
        {
          label: 'New Chat',
          accelerator: 'CmdOrCtrl+N',
          click() {
            win.webContents.executeJavaScript('chatController.clearChat()');
          },
        },
        {
          label: 'Save Chat',
          accelerator: 'CmdOrCtrl+S',
          click() {
            win.webContents.send('save-shortcut-triggered');
          },
        },
        {
          label: 'Stop',
          accelerator: 'Control+C',
          click() {
            win.webContents.executeJavaScript('chatController.requestStopProcess()');
          },
        },
        {
          label: 'Download Chat Logs',
          click() {
            win.webContents.send('download-logs');
          },
        },
        {
          label: 'Check for Updates',
          click: () => {
            if (isUpdateInProgress) return;
            isUpdateInProgress = true;
            autoUpdater
              .checkForUpdates()
              .then((updateCheckResult) => {
                if (updateCheckResult && updateCheckResult.updateAvailable) {
                  win.webContents.executeJavaScript(
                    "viewController.updateFooterMessage('Update available. Downloading...')",
                  );
                } else {
                  isUpdateInProgress = false;
                  win.webContents.executeJavaScript("viewController.updateFooterMessage('App is up to date')");
                }
              })
              .catch((error) => {
                console.error(error);
                win.webContents.executeJavaScript(
                  `viewController.updateFooterMessage('Error occured when updating app. ${error.toString()}')`,
                );
              });
          },
        },
        {
          label: 'Quit',
          accelerator: 'CmdOrCtrl+Q',
          click() {
            app.quit();
          },
        },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'pasteAndMatchStyle' },
        { role: 'delete' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        {
          label: 'Minimize',
          accelerator: 'CmdOrCtrl+M',
          click: () => windowManager.minimize(),
        },
        {
          label: 'Maximize',
          accelerator: 'CmdOrCtrl+Alt+F',
          click: () => windowManager.maximize(),
        },
        {
          label: 'Toggle Full Screen',
          accelerator: 'F11',
          click: () => windowManager.toggleFullScreen(),
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);

  win.on('show', () => {
    win.webContents.executeJavaScript('viewController.onShow()');
    if (!isUpdateInProgress) {
      isUpdateInProgress = true;
      autoUpdater.checkForUpdates();
    }
  });

  win.on('focus', () => {
    win.webContents.executeJavaScript('viewController.onShow()');
  });

  win.on('closed', () => {
    win = null;
  });

  function saveWindowState() {
    const { width, height, x, y } = win.getBounds();
    localStorage.set('windowBounds', {
      width,
      height,
      x,
      y,
    });
    win.webContents.executeJavaScript('chatController.terminalSession.resizeTerminalWindow()');
  }

  win.on(
    'resize',
    debounce(() => {
      saveWindowState();
    }, 200),
  );
  win.on(
    'move',
    debounce(() => {
      saveWindowState();
    }, 200),
  );

  win.webContents.on('did-finish-load', () => {
    const version = app.getVersion();
    const userDataPath = app.getPath('userData');
    win.webContents.send('app-info', { version, userDataPath });
    win.webContents.executeJavaScript('viewController.onShow()');
  });

  // Autoupdater

  autoUpdater.on('checking-for-update', () => {
    win.webContents.executeJavaScript("viewController.updateFooterMessage('Checking for update...')");
  });

  autoUpdater.on('update-available', () => {
    isUpdateInProgress = true;
    win.webContents.executeJavaScript("viewController.updateFooterMessage('Update available. Downloading...')");
  });

  autoUpdater.on('update-not-available', () => {
    isUpdateInProgress = false;
    win.webContents.executeJavaScript("viewController.updateFooterMessage('App is up to date')");
  });

  autoUpdater.on('update-downloaded', () => {
    isUpdateInProgress = true;
    win.webContents.executeJavaScript("viewController.updateFooterMessage('Restart to install updates')");

    dialog
      .showMessageBox({
        type: 'info',
        title: 'Update downloaded',
        message: 'Update downloaded. Would you like to install now?',
        buttons: ['Restart', 'Later'],
      })
      .then((buttonIndex) => {
        if (buttonIndex.response === 0) {
          autoUpdater.quitAndInstall();
        }
      });
  });

  autoUpdater.on('error', (error) => {
    isUpdateInProgress = false;
    console.error(error);
    if (error.message.includes('net::ERR_INTERNET_DISCONNECTED')) {
      win.webContents.executeJavaScript("viewController.updateFooterMessage('Internet connection is not available.')");
    }
  });

  autoUpdater.on('download-progress', (progressObj) => {
    const log_message = `Update downloading ${Math.round(progressObj.percent)}%`;
    win.webContents.executeJavaScript(`viewController.updateFooterMessage('${log_message}')`);
  });
}

app.whenReady().then(() => {
  createWindow();
});

async function openDirectory(sender) {
  try {
    const mainWindow = BrowserWindow.fromWebContents(sender);
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory', 'createDirectory'],
    });

    sender.send('directory-data', { filePaths: result.filePaths });
  } catch (err) {
    console.error(err);
  }
}

async function openFile(sender) {
  try {
    const mainWindow = BrowserWindow.fromWebContents(sender);
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile', 'multiSelections'],
    });

    if (!result.canceled && result.filePaths.length > 0) {
      sender.send('read-files', { filePaths: result.filePaths });
    }
  } catch (err) {
    console.error(err);
  }
}

app.on('window-all-closed', () => {
  autoUpdater.removeAllListeners();

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  const mainWindow = BrowserWindow.getAllWindows()[0];
  if (mainWindow) {
    mainWindow.show();
  } else {
    createWindow();
  }
});

ipcMain.on('open-file-dialog', (event) => {
  openFile(event.sender);
});

ipcMain.on('open-directory', (event) => {
  openDirectory(event.sender);
});

// Shell

ipcMain.on('start-shell', (event, args) => {
  if (terminal) {
    terminal.kill();
    terminal = null;
  }

  const shell = process.platform === 'win32' ? 'powershell.exe' : process.platform === 'darwin' ? 'zsh' : 'bash';
  const shell_args = process.platform === 'win32' ? [] : ['-l'];

  terminal = pty.spawn(shell, shell_args, {
    name: 'xterm-256color',
    cwd: args.cwd,
    env: process.env,
  });

  const shellName = path.basename(shell);
  event.sender.send('shell-type', shellName);

  terminal.on('data', (data) => {
    event.sender.send('shell-data', data);
  });
});

ipcMain.on('kill-shell', () => {
  if (terminal) {
    terminal.kill();
    terminal = null;
  }
});

ipcMain.on('write-shell', (event, args) => {
  if (terminal) {
    terminal.write(args);
  }
});

ipcMain.on('execute-command', (event, command) => {
  let shell, shellArgs;

  if (process.platform === 'win32') {
    shell = 'powershell.exe';
    shellArgs = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', `& {${command}}`];
  } else {
    shell = process.platform === 'darwin' ? '/bin/zsh' : '/bin/bash';
    shellArgs = ['-l', '-c', command];
  }

  const tempTerminal = pty.spawn(shell, shellArgs, {
    name: 'xterm-256color',
    cwd: process.cwd(),
    env: process.env,
  });

  tempTerminal.on('data', (data) => {
    event.sender.send('command-output', data);
  });

  tempTerminal.on('exit', (exitCode) => {
    event.sender.send('command-exit', exitCode);
    tempTerminal.kill();
  });
});

ipcMain.on('resize-shell', (event, data) => {
  if (terminal) {
    terminal.resize(data.cols, data.rows);
  }
});

ipcMain.on('theme-change', (event, theme) => {
  nativeTheme.themeSource = theme === 'dark' ? 'dark' : 'light';
});

app.commandLine.appendSwitch('disable-site-isolation-trials');

// Global error handlers
process.on('uncaughtException', (error) => {
  console.error('Unhandled Error: ', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Promise Rejection: ', promise, ' reason: ', reason);
});
