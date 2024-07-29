const { google } = require('googleapis');
const customsearch = google.customsearch('v1');

const TOP_N_RESULTS = 10;

class GoogleSearch {
  constructor() {
    // This keys are provided out of good will for the community, please do not abuse them or you will get them revoked.
    this.apiKey = 'AIzaSyCcVfWEPZu_CYlQvxDou_sujjgjedSARJk';
    this.cxId = '80a4581942dad4e4d';
  }

  async singleSearch(query) {
    const res = await customsearch.cse.list({
      cx: this.cxId,
      q: query,
      auth: this.apiKey,
    });
    return res.data.items;
  }

  async multipleSearch(queries) {
    const promises = queries.map((query) => this.singleSearch(query));
    const results = await Promise.all(promises);

    return results;
  }

  async search(queries) {
    let formattedResults = [];
    let results = await this.multipleSearch(queries);

    if (results) {
      formattedResults = results.map((result) => {
        if (result && result.length > 0) {
          return result.slice(0, TOP_N_RESULTS).map((item, index) => {
            return {
              relevancy_score: 1 - index / TOP_N_RESULTS,
              title: item.title,
              link: item.link,
              snippet: item.snippet,
            };
          });
        }
        return [];
      });
      formattedResults = formattedResults.flat().sort((a, b) => b.relevancy_score - a.relevancy_score);
      formattedResults = formattedResults
        .filter((item, index, self) => {
          return index === self.findIndex((t) => t.link === item.link);
        })
        .slice(0, TOP_N_RESULTS);
      const rerankedResults = await this.rerankSearchResults(queries, formattedResults);
      return rerankedResults;
    }
  }

  async rerankSearchResults(queries, searchResults) {
    try {
      const searchResultsWithIndex = searchResults.map((result, index) => {
        return { index: index, ...result };
      });
      const prompt = `I am searching internet for this query: '${queries[0]}'.
Search results are:

${JSON.stringify(searchResultsWithIndex)}

What array indexes of these search result objects in the JSON array above are the most relevant and complete search results that may answer the question in my search queries?
Respond with a JSON array containing only the actual array indexes in order of relevance. Minimize the number of items returned, but ensure the results are sufficient to satisfy the research.`;

      const format = {
        type: 'array',
        description: 'Array of indexes representing the most relevant search results',
        items: {
          type: 'integer',
        },
      };

      const rankings = await chatController.backgroundTask.run({ prompt, format });
      const rankedResults = rankings.filter((index) => index in searchResults).map((index) => searchResults[index]);
      return rankedResults;
    } catch (error) {
      console.error(error);
      return searchResults;
    }
  }
}

module.exports = GoogleSearch;
