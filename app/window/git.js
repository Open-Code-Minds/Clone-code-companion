const simpleGit = require('simple-git');
const fs = require('graceful-fs');
const path = require('path');
const { Diff2HtmlUI } = require('diff2html/lib/ui/js/diff2html-ui');

class Git {
  constructor(workingDirectory) {
    this.git = simpleGit(workingDirectory);
    this.workingDirectory = workingDirectory;
    this.selectedFile = null;
    this.updateTabIcon();
  }

  async isGitRepository() {
    try {
      await this.git.status();
      return true;
    } catch (error) {
      return false;
    }
  }

  async updateTabIcon() {
    if (!(await this.isGitRepository())) {
      document.getElementById('gitIcon').innerHTML = '<i class="bi bi-git me-2"></i>';
      return;
    }

    const changedFiles = await this.getChangedFiles();
    document.getElementById('gitIcon').innerHTML =
      `<span class="badge bg-primary rounded-pill me-2">${changedFiles.length}</span>`;
  }

  async getChangedFiles() {
    const status = await this.git.status();
    return [
      ...status.modified.map((file) => ({ file, status: 'modified' })),
      ...status.not_added.map((file) => ({ file, status: 'not_added' })),
      ...status.deleted.map((file) => ({ file, status: 'deleted' })),
    ];
  }

  async getDiff() {
    const status = await this.git.status();
    let diff = await this.git.diff();

    // Add untracked files to the diff
    for (const file of status.not_added) {
      const filePath = path.join(this.workingDirectory, file);
      const content = await fs.promises.readFile(filePath, 'utf8');
      diff += `\ndiff --git a/${file} b/${file}\nnew file mode 100644\nindex 0000000..${Buffer.from(content).toString('hex').slice(0, 7)}\n--- /dev/null\n+++ b/${file}\n@@ -0,0 +1,${content.split('\n').length} @@\n${content
        .split('\n')
        .map((line) => `+${line}`)
        .join('\n')}`;
    }

    return diff;
  }

  async commit() {
    const message = document.getElementById('commit-message').value;
    try {
      await this.git.add('.');
      const result = await this.git.commit(message);
      await this.renderUI();
    } catch (error) {
      console.error(`Error committing changes:`, error);
    }
  }

  async pull() {
    await this.git.pull();
    await this.renderUI();
  }

  async push() {
    await this.git.push();
    await this.renderUI();
  }

  async discardChange(file) {
    try {
      await this.git.checkout([file]);
      await this.renderUI();
      return true;
    } catch (error) {
      console.error(`Error discarding changes for ${file}:`, error);
      return false;
    }
  }

  async showChanges(selectedFile = null) {
    if (!(await this.isGitRepository())) {
      return;
    }

    const diffConfig = {
      drawFileList: false,
      highlight: true,
      matching: 'lines',
      colorScheme: chatController.settings.theme,
      showDiffOnly: false,
      fileContentToggle: true,
    };

    let diff;
    if (selectedFile) {
      diff = await this.git.diff([selectedFile]);
      if (!diff) {
        // If diff is empty, it might be a new file
        const status = await this.git.status();
        const fileStatus = status.files.find((f) => f.path === selectedFile);
        if (fileStatus && fileStatus.index === '?' && fileStatus.working_dir === '?') {
          // It's a new file, so we need to show its content
          const content = await fs.promises.readFile(selectedFile, 'utf8');
          diff = `diff --git a/${selectedFile} b/${selectedFile}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/${selectedFile}\n@@ -0,0 +1,${content.split('\n').length} @@\n${content
            .split('\n')
            .map((line) => '+' + line)
            .join('\n')}`;
        }
      }
    } else {
      diff = await this.getDiff();
    }

    if (!diff) {
      document.getElementById('diff-display').innerHTML =
        '<p class="text-secondary text-center mt-5">No changes to display</p>';
      return;
    }

    const targetElement = document.getElementById('diff-display');
    const diff2htmlUi = new Diff2HtmlUI(targetElement, diff, diffConfig);
    diff2htmlUi.draw();
    diff2htmlUi.highlightCode();
  }

  async renderUI() {
    let html = '';
    let changedFiles = [];
    if (await this.isGitRepository()) {
      changedFiles = await this.getChangedFiles();
      const branchName = await this.git.branch();
      const commitButtonDisabled = changedFiles.length === 0;
      html = `
        <div id="git-commit-ui" class="container-fluid">
          <div class="row">
            <div class="col-md-3 border-end py-3">
              <div class="d-flex justify-content-end mb-1">
                ${this.renderActions()}
              </div>
              <input id="commit-message" class="form-control mb-3" placeholder="Message for &quot;${branchName.current}&quot;">
              <button id="commit-button" class="btn btn-primary w-100 ${commitButtonDisabled ? 'disabled' : ''}" onclick="chatController.agent.projectController.git?.commit();">
                <i class="bi bi-check2-all"></i>
                Commit all
              </button>
              <div class="mt-3 text-secondary d-flex align-items-center justify-content-between">
                Changes
                <span class="badge rounded-pill bg-primary ms-2">${changedFiles.length}</span>
              </div>
              <ul class="list-group list-group-flush mt-2">
                ${changedFiles.map((file) => this.renderFileItem(file)).join('')}
              </ul>
            </div>
            <div class="col-md-9 py-3">
              <div id="diff-display" class="overflow-y-auto overflow-x-hidden" style="height: calc(100vh - 150px);"></div>
            </div>
          </div>
        </div>
      `;
    } else {
      html = `
        <div class="d-flex justify-content-center align-items-center h-100">
          <p class="text-secondary text-center">
            Open a project with a Git repository to use Git features.
            <br>
            Or ask in chat to initialize a new repository.
          </p>
        </div>
      `;
    }

    viewController.activateTab('git-tab');
    document.getElementById('git_output').innerHTML = html;
    viewController.activateTooltips();
    await this.showChanges();
    this.updateTabIcon();
  }

  renderActions() {
    return `
      <button
        id="git-refresh"
        class="btn btn-link-secondary p-0 ms-1"
        data-bs-toggle="tooltip"
        data-bs-title="Refresh"
        onclick="chatController.agent.projectController.git?.renderUI();"
      >
        <i class="bi bi-arrow-clockwise"></i>
      </button>
    `;
  }

  renderFileItem(file) {
    const normalizedPath = path.normalize(file.file);
    const escapedPath = normalizedPath.replace(/"/g, '&quot;');
    return `
      <li class="list-group-item d-flex justify-content-between align-items-center p-0 ps-1" data-file="${escapedPath}">
        <span class="text-truncate mw-75 file-name" style="cursor: pointer;" onclick="chatController.agent.projectController.git?.showFileChanges('${escapedPath}');">${path.basename(normalizedPath)}</span>
        <span class="ms-2 text-nowrap align-items-center d-flex">
          <span class="text-${file.status === 'modified' ? 'info' : file.status === 'not_added' ? 'success' : 'danger'} me-2 d-inline-block text-center">${file.status.charAt(0).toUpperCase()}</span>
          <button class="btn btn-link-secondary p-0" data-bs-toggle="tooltip" data-bs-title="Discard changes" onclick="chatController.agent.projectController.git?.discardChange('${escapedPath}');">
            <i class="bi bi-x-circle"></i>
          </button>
        </span>
      </li>
    `;
  }

  async showFileChanges(file) {
    if (this.selectedFile === file) {
      this.selectedFile = null;
      await this.showChanges();
      this.setActiveFile(null);
    } else {
      this.selectedFile = file;
      await this.showChanges(file);
      this.setActiveFile(file);
    }
  }

  setActiveFile(file) {
    const fileItems = document.querySelectorAll('#git_output .list-group-item');
    fileItems.forEach((item) => {
      if (file === null || item.dataset.file === file) {
        item.classList.toggle('active', item.dataset.file === file);
      } else {
        item.classList.remove('active');
      }
    });
  }
}

module.exports = Git;
