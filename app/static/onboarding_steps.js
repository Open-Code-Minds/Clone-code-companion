module.exports = [
  {
    id: 'missing_api_key',
    condition: () => !document.getElementById('apiKey').value,
    description: `
      Hello! Welcome to CodeCompanion.AI. To get started: <br /><br />
      <ul class="lh-lg">
        <li>First, add your <a href="https://platform.openai.com/account/api-keys">OpenAI API key</a> in the settings menu (<i class="bi bi-gear border-0"></i>).</li>
        <li>
          By using this app you agree to
          <a href="https://www.codecompanion.ai/terms">Terms</a><br />
          You acknowledge the potential risks of using an AI assistant such
          as CodeCompanion.AI. This may include, but is not limited to, the
          execution of shell commands that could delete or overwrite files,
          destroy data, or harm your computer. <br />
          Excessive use may result in high OpenAI API charges. Monitor your use and set a budget to avoid unexpected costs.
        </li>
      </ul>
      <br />
      Happy coding!
    `,
  },
];
