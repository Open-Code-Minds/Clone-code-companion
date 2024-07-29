const { log } = require('./utils');

const SYSTEM_PROMPT = 'Provide result to user task in tool call in "result" property';

class BackgroundTask {
  //  format example:
  //   {
  //       "type": "string",
  //       "description": "The city and country, eg. San Francisco, USA"
  //   }

  constructor(chatController) {
    this.messages = [];
    this.client = chatController.smallModel;
    this.chatController = chatController;
  }

  async run({ prompt, format }) {
    try {
      const messages = this.buildMessages(prompt);
      const tool = this.buildTool(format);
      log('BackgroundTask:');
      const response = await this.client.call({ messages, tool });
      this.chatController.updateUsage(response.usage);
      return response.content;
    } catch (error) {
      console.error(error);
    }
  }

  buildMessages(prompt) {
    return [
      {
        role: 'system',
        content: SYSTEM_PROMPT,
      },
      {
        role: 'user',
        content: prompt,
      },
    ];
  }

  buildTool(format) {
    return {
      name: 'json_call',
      parameters: {
        type: 'object',
        properties: {
          result: format,
        },
      },
    };
  }
}

module.exports = BackgroundTask;
