const { BrowserWindow } = require('electron');

class WindowManager {
  constructor(window) {
    this.window = window;
  }

  minimize() {
    this.window.minimize();
  }

  maximize() {
    if (this.window.isMaximized()) {
      this.window.unmaximize();
    } else {
      this.window.maximize();
    }
  }

  close() {
    this.window.close();
  }

  toggleFullScreen() {
    this.window.setFullScreen(!this.window.isFullScreen());
  }

  moveWindow(direction) {
    const [x, y] = this.window.getPosition();
    const step = 50;

    switch (direction) {
      case 'left':
        this.window.setPosition(x - step, y);
        break;
      case 'right':
        this.window.setPosition(x + step, y);
        break;
      case 'up':
        this.window.setPosition(x, y - step);
        break;
      case 'down':
        this.window.setPosition(x, y + step);
        break;
    }
  }

  resizeWindow(direction) {
    const [width, height] = this.window.getSize();
    const step = 50;

    switch (direction) {
      case 'wider':
        this.window.setSize(width + step, height);
        break;
      case 'narrower':
        this.window.setSize(Math.max(width - step, 200), height);
        break;
      case 'taller':
        this.window.setSize(width, height + step);
        break;
      case 'shorter':
        this.window.setSize(width, Math.max(height - step, 200));
        break;
    }
  }
}

module.exports = WindowManager;