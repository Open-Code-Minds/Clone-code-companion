async function contextualCompress(query, text) {
  const prompt = `
  Given the following query and text, extract the most relevant information that directly answers or relates to the query. Do not modify or paraphrase the extracted information. Maintain the original wording and context.

  Query: ${query}

  <text>
  ${text}
  </text>

  Instructions:
  1. Identify the parts of the text that are most relevant to the query.
  2. Extract these relevant parts verbatim, without any modifications or paraphrasing.
  3. If there are multiple relevant sections, include all of them.
  4. Preserve the original context and wording of the extracted information.
  5. If no information in the text is directly relevant to the query, respond with "No relevant information found."

  Relevant information (extracted verbatim):
  `;
  const format = {
    type: 'string',
    result: 'Extracted information',
  };
  const result = await chatController.backgroundTask.run({
    prompt,
    format,
    model: chatController.settings.selectedModel,
  });
  return result ? result : text;
}

module.exports = {
  contextualCompress,
};
