const { OpenAI } = require('openai');
const { log, getTokenCount } = require('../utils');

const MAX_RETRIES = 5;

class OpenRouterModel {
  constructor({ model, apiKey, streamCallback, chatController }) {
    this.model = model;
    this.chatController = chatController;
    const config = {
      apiKey: apiKey,
      dangerouslyAllowBrowser: true,
      maxRetries: MAX_RETRIES,
      baseURL: 'https://openrouter.ai/api/v1',
    };
    this.client = new OpenAI(config);
    this.streamCallback = streamCallback;
  }

  async call({ messages, model, tools = null, temperature = 0.0 }) {
    const callParams = {
      model: model || this.model,
      messages,
      temperature,
    };
    if (tools) {
      callParams.tools = tools.map((tool) => this.openAiToolFormat(tool));
      callParams.tool_choice = "auto";
    }
    return await this.stream(callParams);
  }

  async stream(callParams) {
    callParams.stream = true;
    log('Calling OpenRouter API:', callParams);
    const stream = this.client.beta.chat.completions.stream(callParams, {
      signal: this.chatController.abortController.signal,
    });
    let fullContent = '';
    let toolCalls = [];
    
    stream.on('content', (_delta, snapshot) => {
      fullContent = snapshot;
      this.streamCallback(snapshot);
    });

    stream.on('tool_call', (toolCall) => {
      toolCalls.push(toolCall);
    });

    const chatCompletion = await stream.finalChatCompletion();
    log('Raw response', chatCompletion);

    return {
      content: fullContent,
      tool_calls: this.formattedToolCalls(toolCalls),
      usage: {
        input_tokens: getTokenCount(callParams.messages),
        output_tokens: getTokenCount(chatCompletion.choices[0].message),
      },
    };
  }


  formattedToolCalls(toolCalls) {
    if (!toolCalls || toolCalls.length === 0) return null;

    return toolCalls.map(toolCall => ({
      id: toolCall.id,
      type: 'function',
      function: {
        name: toolCall.function.name,
        arguments: toolCall.function.arguments
      }
    }));
  }

  openAiToolFormat(tool) {
    return {
      type: 'function',
      function: tool,
    };
  }

  abort() {
    this.chatController.abortController.abort();
    this.chatController.abortController = new AbortController();
  }
}

module.exports = OpenRouterModel;
