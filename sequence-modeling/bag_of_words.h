#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <unordered_set>

using namespace std;

/** Tokenizes text ignoring stop-words */
inline vector<string> tokenize(
    const string& text,
    const unordered_set<string>& stop_words
) {
    vector<string> tokens;
    stringstream ss(text);
    string word;

    while (ss >> word) {
        // Lowervase normalization
        transform(word.begin(), word.end(), word.end(), ::tolower);

        // Remove basic punctuation
        word.erase(remove_if(word.begin(), word.end(), [](char c) {
            return ispunct(static_cast<unsigned char>(c));
        }), word.end());

        if (!word.empty() && stop_words.find(word) == stop_words.end()) {
            tokens.push_back(word);
        }
    }

    return tokens;
}


inline vector<string> buildVocabulary(
    const vector<string>& documents,
    const unordered_set<string>& stop_words

) {
    unordered_set<string> vocabSet;

    for (const string& doc : documents) {
        auto tokens = tokenize(doc, stop_words);
        for (const string& token : tokens) {
            vocabSet.insert(token);
        }
    }

    vector<string> vocabulary(vocabSet.begin(), vocabSet.end());
    sort(vocabulary.begin(), vocabulary.end());

    return vocabulary;
}


inline vector<vector<int>> bagOfWords(
    const vector<string>& documents,
    const vector<string>& vocabulary,
    const unordered_set<string>& stop_words
) {
    unordered_map<string, int> wordIndex;

    for (size_t i = 0; i < vocabulary.size(); i++) {
        wordIndex[vocabulary[i]] = i;
    }

    vector<vector<int>> bow_matrix(
        documents.size(),
        vector<int>(vocabulary.size(), 0)
    );

    for (int i = 0; i < (int)documents.size(); i++) {
        vector<string> tokens = tokenize(documents[i], stop_words);

        for (const string& token : tokens) {
            if (wordIndex.count(token)) {
                bow_matrix[i][wordIndex[token]]++;
            }
        }
    }

    return bow_matrix;
}
