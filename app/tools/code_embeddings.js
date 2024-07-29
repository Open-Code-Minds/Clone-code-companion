const fs = require('graceful-fs');
const pathModule = require('path');
const CryptoJS = require('crypto-js');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const detect = require('language-detect');
const { normalizedFilePath } = require('../utils');

const { isTextFile } = require('../utils');
const { EMBEDDINGS_VERSION, EMBEDDINGS_MODEL_NAME } = require('../static/models_config');

const detectedLanguageToSplitterMapping = {
  'C++': 'cpp',
  Go: 'go',
  Java: 'java',
  JavaScript: 'js',
  PHP: 'php',
  'Protocol Buffers': 'proto',
  Python: 'python',
  reStructuredText: 'rst',
  Ruby: 'ruby',
  Rust: 'rust',
  Scala: 'scala',
  Swift: 'swift',
  Markdown: 'markdown',
  LaTeX: 'latex',
  HTML: 'html',
  Solidity: 'sol',
};

const MAX_FILE_SIZE = 30000;

class CodeEmbeddings {
  constructor(projectName, openAIApiKey) {
    this.projectName = projectName;
    this.openAIApiKey = openAIApiKey;
    this.vectorStore = new MemoryVectorStore(
      new OpenAIEmbeddings({
        openAIApiKey,
        modelName: EMBEDDINGS_MODEL_NAME,
        maxRetries: 5,
        timeout: 5 * 60 * 1000,
      }),
    );
  }

  async splitCodeIntoChunks(metadata, fileContent, language) {
    let splitter;
    if (!language || language === 'other') {
      splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
        keepSeparator: true,
      });
    } else {
      splitter = RecursiveCharacterTextSplitter.fromLanguage(language, {
        chunkSize: 1000,
        chunkOverlap: 0,
        keepSeparator: true,
      });
    }
    const documents = await splitter.createDocuments([fileContent], [metadata], {
      chunkHeader: `File name: ${metadata.filePath}\n---\n\n`,
      appendChunkOverlapHeader: true,
    });
    return documents;
  }

  async updateEmbeddingsForFiles(filesList) {
    if (!this.openAIApiKey) {
      return;
    }

    if (filesList.length === 0) {
      this.deleteEmbeddingsForFilesNotInList(filesList);
      this.save();
      return;
    }

    const filesNeedingReembedding = (
      await Promise.all(
        filesList.map(async (filePath) => {
          return (await this.needsReembedding(filePath)) ? filePath : null;
        }),
      )
    ).filter((filePath) => filePath !== null);

    viewController.updateLoadingIndicator(
      true,
      `Indexing ${filesNeedingReembedding.length} files with vector embeddings...`,
    );
    const promises = filesNeedingReembedding.map((file) => this.updateEmbedding(file));
    await Promise.all(promises);
    this.deleteEmbeddingsForFilesNotInList(filesList);
    this.save();
  }

  async needsReembedding(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const stats = await fs.promises.stat(filePath);

    if (!isTextFile(filePath) || stats.size > MAX_FILE_SIZE) {
      return false;
    }

    const hash = CryptoJS.SHA256(fileContent).toString() + EMBEDDINGS_VERSION;
    const fileRecords = this.findRecords(filePath);
    if (fileRecords.length === 0) return true;

    return fileRecords[0].metadata.hash !== hash;
  }

  async updateEmbedding(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');

    if (!isTextFile(filePath)) {
      console.log('skipping embedding for', filePath);
      return;
    }

    const hash = CryptoJS.SHA256(fileContent).toString() + EMBEDDINGS_VERSION;
    this.deleteRecords(filePath);

    const metadata = {
      filePath,
      hash,
    };

    let language;
    try {
      language = detect.sync(filePath);
    } catch (error) {
      // ignore
    }

    const mappedLanguage = detectedLanguageToSplitterMapping[language] || 'other';
    const documents = await this.splitCodeIntoChunks(metadata, fileContent, mappedLanguage);
    if (documents && documents.length > 0) {
      await this.vectorStore.addDocuments(documents);
    }
  }

  isEmbededAndCurrent(filePath, hash) {
    const records = this.findRecords(filePath);
    if (records.length === 0) return false;

    return records[0].metadata.hash === hash;
  }

  deleteEmbeddingsForFilesNotInList(filesList) {
    const filePathsToKeep = new Set(filesList);
    this.vectorStore.memoryVectors = this.vectorStore.memoryVectors.filter((record) =>
      filePathsToKeep.has(record.metadata.filePath),
    );
  }

  findRecords(filePath) {
    return this.vectorStore.memoryVectors.filter((record) => record.metadata.filePath === filePath);
  }

  deleteRecords(filePath) {
    this.vectorStore.memoryVectors = this.vectorStore.memoryVectors.filter(
      (record) => record.metadata.filePath !== filePath,
    );
  }

  async search({ query, limit = 50, basePath, minScore = 0.4, rerank = true, filenamesOnly = false }) {
    const results = await this.vectorStore.similaritySearchWithScore(query, limit * 2);
    if (!results) return [];

    if (filenamesOnly) {
      const filePaths = results.map((result) => {
        const [record, _score] = result;
        return normalizedFilePath(record.metadata.filePath);
      });
      return Promise.all(filePaths).then((paths) => [...new Set(paths)].slice(0, limit));
    }

    const filteredResults = results.filter((result) => {
      const [record, score] = result;
      return score >= minScore && record.pageContent.length > 5;
    });

    const formattedResults = await Promise.all(
      filteredResults.map(async (result) => {
        const [record, _score] = result;
        return {
          filePath: await normalizedFilePath(record.metadata.filePath),
          fileContent: record.pageContent,
          lines: record.metadata.loc.lines,
        };
      }),
    );

    if (!rerank) {
      return formattedResults.slice(0, limit);
    }

    const rerankedResults = await this.rerankSearchResults(query, formattedResults);
    if (rerankedResults && rerankedResults.length > 0) {
      return rerankedResults.slice(0, limit);
    }

    return formattedResults.slice(0, limit);
  }

  async rerankSearchResults(query, searchResults) {
    try {
      const searchResultsWithIndex = searchResults.map((result, index) => {
        return { index: index, filePath: result.filePath, fileContent: result.fileContent };
      });

      const prompt = `Codebase search query: "${query}"\n
Codebase snippets search results:

${JSON.stringify(searchResultsWithIndex)}

What array indexes of these search result objects in the JSON array above are the most relevant to my search query?
Respond with a JSON array containing only the actual array indexes in order of search result relevance. Do not include indexes of totally irrelevant search results.`;
      const format = {
        type: 'array',
        description: 'Array of indexes representing the most relevant search results',
        items: {
          type: 'integer',
        },
      };

      const parsedRankings = await chatController.backgroundTask.run({ prompt, format });
      const rankedResults = parsedRankings
        .filter((index) => index in searchResults)
        .map((index) => searchResults[index]);
      return rankedResults;
    } catch (error) {
      return searchResults;
    }
  }

  save() {
    localStorage.set(`project.${this.projectName}.embeddings`, JSON.stringify(this.vectorStore.memoryVectors));
  }

  async load() {
    const serializedVectors = localStorage.get(`project.${this.projectName}.embeddings`);
    if (!serializedVectors) return;

    const vectors = JSON.parse(serializedVectors);
    this.vectorStore.memoryVectors = vectors;
  }
}

module.exports = CodeEmbeddings;
