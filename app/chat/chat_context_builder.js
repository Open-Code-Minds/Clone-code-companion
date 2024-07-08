const ignore = require('ignore');

const {
  PLAN_PROMPT_TEMPLATE,
  TASK_EXECUTION_PROMPT_TEMPLATE,
  FINISH_TASK_PROMPT_TEMPLATE,
} = require('../static/prompts');
const { withErrorHandling, getSystemInfo, isTextFile } = require('../utils');
const { normalizedFilePath } = require('../utils');
const ignorePatterns = require('../static/embeddings_ignore_patterns');

const MAX_SUMMARY_TOKENS = 1000;
const MAX_RELEVANT_FILES_TOKENS = 10000;
const MAX_RELEVANT_FILES_COUNT = 6;
const MAX_FILE_SIZE = 30000;
const SUMMARIZE_MESSAGES_THRESHOLD = 4; // Last n message will be left as is

class ChatContextBuilder {
  constructor(chat) {
    this.chat = chat;
    this.lastSummarizedMessageID = 0;
    this.lastMessageIdForRelevantFiles = 0;
    this.reduceRelevantFilesContextMessageId = 0;
    this.lastEditedFilesTimestamp = chat.startTimestamp;
    this.taskRelevantFiles = [];
    this.pastSummarizedMessages = '';
    this.searchRelevantFiles = false;
    this.taskNeedsPlan = false;
  }

  async buildMessages(userMessage, reflectMessage = null) {
    this.backendMessages = this.chat.backendMessages.map((message) => _.omit(message, ['id']));

    return [await this.addSystemMessage(), await this.addUserMessage(userMessage, reflectMessage)];
  }

  async addUserMessage(userMessage, reflectMessage) {
    const conversationSummary = await this.addSummaryOfMessages();
    const lastUserMessage = this.addLastUserMessage(userMessage);
    const reflectMessageResult = this.addReflectMessage(reflectMessage);
    const relevantSourceCodeInformation = await this.relevantSourceCodeInformation();
    const content = [
      this.addTaskMessage(),
      conversationSummary,
      lastUserMessage,
      relevantSourceCodeInformation,
      reflectMessageResult,
    ]
      .filter(Boolean)
      .join('\n');
    return {
      role: 'user',
      content,
    };
  }

  addImageMessages() {
    const imageMessages = this.backendMessages.filter(
      (message) => Array.isArray(message.content) && message.content.some((content) => content.type === 'image_url'),
    );

    return imageMessages;
  }

  async addSystemMessage() {
    let systemMessage;

    if (
      (this.taskNeedsPlan && this.chat.countOfUserMessages() === 0) ||
      (this.chat.isEmpty() && (await this.isTaskNeedsPlan()))
    ) {
      this.taskNeedsPlan = true;
      systemMessage = PLAN_PROMPT_TEMPLATE;
    } else {
      this.taskNeedsPlan = false;
      systemMessage = TASK_EXECUTION_PROMPT_TEMPLATE;
    }

    if (this.chat.backendMessages.length > 7) {
      systemMessage += `\n\n${FINISH_TASK_PROMPT_TEMPLATE}`;
    }

    systemMessage += this.addProjectCustomInstructionsMessage();
    systemMessage = this.fromTemplate(systemMessage, '{osName}', getSystemInfo());
    systemMessage = this.fromTemplate(systemMessage, '{shellType}', chatController.terminalSession.shellType);

    return {
      role: 'system',
      content: systemMessage,
    };
  }

  async isTaskNeedsPlan() {
    const prompt = `
    Task:
    "${this.chat.task}"\n
    Is this user task will need to be brainstormed and planned before execution? Respond false, if this is a simple task that involves only a few commands or one file manipulation.`;

    const format = {
      type: 'boolean',
      result: 'true or false',
    };

    const result = await chatController.backgroundTask.run({
      prompt,
      format,
      model: chatController.settings.selectedModel,
    });

    return result;
  }

  addTaskMessage() {
    return `<task>\n${this.chat.task}\n</task>\n`;
  }

  addProjectCustomInstructionsMessage() {
    const projectCustomInstructions = chatController.agent.projectController.getCustomInstructions();
    if (!projectCustomInstructions) {
      return '';
    } else {
      return `\n\n${projectCustomInstructions}`;
    }
  }

  async addSummaryOfMessages() {
    let allMessages = '';
    const nonEmptyMessages = this.chat.backendMessages.filter((message) => message.content);
    const preprocessedMessages = nonEmptyMessages.map((message) => {
      if (Array.isArray(message.content) && message.content.some((content) => content.type === 'image_url')) {
        return { ...message, content: 'image added to chat' };
      }
      return message;
    });
    const messagesToSummarize = preprocessedMessages.slice(0, -SUMMARIZE_MESSAGES_THRESHOLD);
    const notSummarizedMessages = messagesToSummarize
      .filter((message) => message.id > this.lastSummarizedMessageID)
      .reduce((acc, message) => {
        if (message.content) {
          let messageContent = message.content;
          const role = message.role == 'tool' ? `assistant` : message.role;
          const tool = message.role == 'tool' ? ` executed_tool="${message.name}"` : '';
          acc += `\n<${role}${tool}>\n${messageContent}\n</${role}>\n`;
        }
        return acc;
      }, '')
      .trim();

    allMessages = this.pastSummarizedMessages + '\n\n' + notSummarizedMessages;

    if (
      this.chat.countTokens(allMessages) > MAX_SUMMARY_TOKENS &&
      this.chat.backendMessages.length > SUMMARIZE_MESSAGES_THRESHOLD * 2
    ) {
      this.pastSummarizedMessages = await this.summarizeMessages(allMessages);
      // Update last summarized message ID to the second last message if messages were summarized
      this.lastSummarizedMessageID =
        messagesToSummarize.length > 0
          ? messagesToSummarize[messagesToSummarize.length - 1].id
          : this.lastSummarizedMessageID;
      allMessages = this.pastSummarizedMessages;
    }

    const lastNMessages = nonEmptyMessages.slice(-SUMMARIZE_MESSAGES_THRESHOLD);
    let messagesToAdd = lastNMessages.filter((message) => message.id > this.lastSummarizedMessageID);
    if (messagesToAdd.length > 0 && messagesToAdd[messagesToAdd.length - 1].role === 'user') {
      messagesToAdd.pop(); // Remove the last message if it's from a user
    }
    messagesToAdd.forEach((message) => {
      let messageContent = message.content;
      const role = message.role == 'tool' ? `assistant` : message.role;
      const tool = message.role == 'tool' ? ` executed_tool="${message.name}"` : '';
      allMessages += `\n<${role}${tool}>\n${messageContent}\n</${role}>\n`;
    });

    const summary =
      allMessages.trim().length > 0 ? `\n<conversation_history>\n${allMessages}\n</conversation_history>` : '';

    return summary;
  }

  async summarizeMessages(messages) {
    const prompt = `
    <task>
    ${this.chat.task}
    </task>
    <messages>
    ${messages}\n
    </messages>
    <instructions>
    Compress the messages above in <messages> section. Preserve the meaning, file names, results, order of actions, what was done and what is left.
    Also preserve any important information or code snippets.
    Leave user's messages and task plan as is word for word. 
    For each type of user leave user type (assistant or user) and summary of the messages for that section of the conversation.
    Use xml syntax to indicate roles and messages and task plan. Example:
    <task_plan>
    Plan
    </task_plan>
    <user>
    User message
    </user>
    Summary should be formatted as text with roles and messages separated by new line.
    </instructions>
    `;
    const format = {
      type: 'string',
      result: 'Summary of the conversation',
    };
    const summary = await chatController.backgroundTask.run({
      prompt,
      format,
      model: chatController.settings.selectedModel,
    });

    if (summary) {
      return JSON.stringify(summary);
    } else {
      return messages;
    }
  }

  async relevantSourceCodeInformation() {
    const projetState = await this.projectStateToText();
    const relevantFilesAndFoldersToUserMessages = await this.getRelevantFilesAndFoldersToUserMessages();
    const relevantFilesContents = await this.getRelevantFilesContents();

    return `${projetState}${relevantFilesAndFoldersToUserMessages}${relevantFilesContents}`;
  }

  async getRelevantFilesAndFoldersToUserMessages() {
    if (!this.searchRelevantFiles) {
      return '';
    }

    let lastBackendMessage = this.chat.backendMessages[this.chat.backendMessages.length - 1];
    let lastUserMessage;
    if (!lastBackendMessage) {
      lastUserMessage = this.chat.task;
    } else {
      if (lastBackendMessage.role === 'user') {
        lastUserMessage = lastBackendMessage;
      }
    }

    if (!lastUserMessage) {
      return '';
    }

    const params = {
      query: this.chat.task + (lastUserMessage ? ' ' + lastUserMessage.content : ''),
      limit: 10,
      filenamesOnly: true,
    };
    const projectController = chatController.agent.projectController;
    if (!projectController.currentProject) {
      return '';
    }

    const relevantFilesAndFolders = await projectController.searchEmbeddings(params);
    if (!relevantFilesAndFolders || relevantFilesAndFolders.length === 0) {
      return '';
    } else {
      const relevantFilesAndFoldersMessage = relevantFilesAndFolders
        .map((result) => {
          return `- "${result}"`;
        })
        .join('\n');
      return `These files might or might not be relevant to the task:\n<relevant_files_and_folders>\n${relevantFilesAndFoldersMessage}\n</relevant_files_and_folders>\n`;
    }
  }

  async getRelevantFilesContents() {
    const relevantFileNames = await this.getListOfRelevantFiles();
    if (relevantFileNames.length === 0) {
      return '';
    }

    let fileContents = await this.getFileContents(relevantFileNames);
    fileContents = await this.reduceRelevantFilesContext(fileContents, relevantFileNames);

    return fileContents
      ? `\n\nCurrent content of the files (no need to read these files again and dont say thank you for providing these files):\n<relevant_files_contents>${fileContents}\n</relevant_files_contents>`
      : '';
  }

  async getListOfRelevantFiles() {
    const chatInteractionFiles = await this.getChatInteractionFiles();
    const editedFiles = chatController.agent.projectController.getRecentModifiedFiles(this.lastEditedFilesTimestamp);
    this.lastEditedFilesTimestamp = Date.now();
    const combinedFiles = [...new Set([...chatInteractionFiles, ...this.taskRelevantFiles, ...editedFiles])].slice(
      0,
      20,
    );
    this.taskRelevantFiles = combinedFiles;

    return combinedFiles;
  }

  async getChatInteractionFiles() {
    const chatFiles = this.chat.backendMessages
      .filter((message) => message.id > this.lastMessageIdForRelevantFiles)
      .filter((message) => message.role === 'assistant' && message.tool_calls)
      .flatMap((message) =>
        message.tool_calls
          .map((toolCall) => {
            const parsedArguments = chatController.agent.parseArguments(toolCall.function.arguments);
            return parsedArguments.hasOwnProperty('targetFile') ? parsedArguments.targetFile : undefined;
          })
          .filter((file) => file !== undefined),
      );
    const normalizedFilePaths = await Promise.all(chatFiles.map((file) => normalizedFilePath(file)));
    const chatInteractionFiles = normalizedFilePaths
      .filter((file) => fs.existsSync(file) && !fs.statSync(file).isDirectory())
      .reverse();
    this.lastMessageIdForRelevantFiles = this.backendMessages.length - 1;

    return chatInteractionFiles;
  }

  async getFileContents(fileList) {
    if (fileList.length === 0) {
      return '';
    }

    const fileReadPromises = fileList.map((file) => this.readFile(file));
    const fileContents = await Promise.all(fileReadPromises);

    return fileList
      .map((file, index) => `\n<file_content file="${file}">\n${fileContents[index]}\n</file_content>`)
      .join('\n\n');
  }

  async reduceRelevantFilesContext(fileContents, fileList) {
    const fileContentTokenCount = this.chat.countTokens(fileContents);
    const lastMessageId = this.chat.backendMessages.length - 1;
    if (
      fileContentTokenCount > MAX_RELEVANT_FILES_TOKENS &&
      fileList.length > MAX_RELEVANT_FILES_COUNT &&
      (lastMessageId - this.reduceRelevantFilesContextMessageId >= 10 || this.reduceRelevantFilesContextMessageId === 0)
    ) {
      this.reduceRelevantFilesContextMessageId = lastMessageId;
      const relevantFiles = await this.updateListOfRelevantFiles(fileContents);
      if (Array.isArray(relevantFiles)) {
        console.log('Reducing relevant files context', relevantFiles);
        this.taskRelevantFiles = relevantFiles.slice(0, 10);
        return await this.getFileContents(relevantFiles);
      }
    }

    return fileContents;
  }

  async updateListOfRelevantFiles(fileContents) {
    const messageHistory = [this.addTaskMessage(), await this.addSummaryOfMessages()];

    const prompt = `AI coding assistnant is helping user with a task.
    Here is summary of the conversation and what was done: ${messageHistory}
    
    The content of the files is too long to process. Out of the list of files below, select the most relevant files that assistant still needs to know contents of in order to complete user's task.
    The files are:\n\n${fileContents}
    
    Include only required files, exclude files that are already processed or most likely not needed.
    Respond with the array of file paths exactly as they appeared (do not shorten or change file path) in the list above separated by comma.
    If all files are relevant, respond with a list of all files.
    Order files by importance, most important first.
    `;

    const format = {
      type: 'array',
      description: 'Array of relevant file paths',
      items: {
        type: 'string',
      },
    };

    const result = await chatController.backgroundTask.run({
      prompt,
      format,
      model: chatController.settings.selectedModel,
    });

    return result;
  }

  async readFile(filePath) {
    try {
      const stats = await fs.promises.stat(filePath);
      if (!isTextFile(filePath) || stats.size > MAX_FILE_SIZE) {
        console.error(`Skipped file (non-text or too large): ${filePath}`);
        return null;
      }
      const content = await fs.promises.readFile(filePath, 'utf8');
      return this.addLineNumbers(content);
    } catch (error) {
      console.error(`Error reading file ${filePath}:`, error);
      return null;
    }
  }

  addLineNumbers(content) {
    const lines = content.split('\n');
    const paddedLines = lines.map((line, index) => {
      const lineNumber = (index + 1).toString().padStart(4, ' ');
      return `${lineNumber} | ${line}`;
    });
    content = paddedLines.join('\n');
    return content;
  }

  addLastUserMessage(userMessage) {
    if (!userMessage) {
      userMessage = '';
    }

    return userMessage ? `<user>${userMessage}</user>\n` : '';
  }

  addReflectMessage(reflectMessage) {
    if (!reflectMessage) {
      return null;
    }

    return `
      "assistant" proposed this change:
      
      ${JSON.stringify(reflectMessage, null, 2)}
      
      This can be improved.
      First step by step explain how code/commamnd can be improved or fixed, what bugs it has, what was not implemented correctly or fully, and what may not work.
      Then run the same tool but with improved code/command based on your explanation. Only provide code/command in the tool not in message content`;
  }

  fromTemplate(content, placeholder, value) {
    const regex = new RegExp(placeholder, 'g');
    return content.replace(regex, value);
  }

  async projectStateToText() {
    const dirName = path.basename(await chatController.terminalSession.getCurrentDirectory());

    let projectStateText = '';
    projectStateText += `Current directory is '${dirName}'. The full path to this directory is '${chatController.agent.currentWorkingDir}'`;
    if (chatController.agent.projectController.currentProject) {
      const filesInFolder = await withErrorHandling(this.getFolderStructure.bind(this));
      if (filesInFolder) {
        projectStateText += `\nThe contents of this directory (excluding files from .gitignore): \n${filesInFolder}`;
      }
    }

    return projectStateText ? `\n<current_project_state>\n${projectStateText}\n</current_project_state>\n` : '';
  }

  async getFolderStructure() {
    const ig = ignore().add(ignorePatterns);
    const rootDir = chatController.agent.currentWorkingDir;

    // Recursive function to list files
    const listFiles = async (dir, allFiles = [], currentPath = '') => {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      for (let entry of entries) {
        const entryPath = path.join(dir, entry.name);
        const relativePath = path.join(currentPath, entry.name);

        if (entry.isDirectory()) {
          await listFiles(entryPath, allFiles, relativePath);
        } else {
          allFiles.push(relativePath);
        }
      }
      return allFiles;
    };

    try {
      const allFiles = await listFiles(rootDir);
      const filesExcludingIgnored = allFiles.filter((file) => !ig.ignores(file));

      if (allFiles.length === 0) {
        // If directory is empty
        this.searchRelevantFiles = false;
        return 'The directory is empty.';
      } else if (filesExcludingIgnored.length <= 30) {
        this.searchRelevantFiles = false;
        return filesExcludingIgnored.map((file) => `- ${file}`).join('\n');
      } else {
        // If more than 30 files, only show top-level directories and files
        this.searchRelevantFiles = true;
        const topLevelEntries = await fs.promises.readdir(rootDir, { withFileTypes: true });
        const filteredTopLevelEntries = topLevelEntries.filter((entry) => !ig.ignores(entry.name));
        const folderStructure = filteredTopLevelEntries
          .map((entry) => `- ${entry.name}${entry.isDirectory() ? '/' : ''}`)
          .join('\n');
        return folderStructure;
      }
    } catch (error) {
      chatController.chat.addFrontendMessage(
        'error',
        `Error occurred while checking directory structure in ${rootDir}.
       <br>Please change directory where app can read/write files or update permissions for current directory.`,
      );
      return;
    }
  }
}

module.exports = ChatContextBuilder;
