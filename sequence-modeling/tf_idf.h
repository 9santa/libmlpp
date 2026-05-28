#include "bag_of_words.h"
#include <cmath>


inline vector<vector<double>> computeTF(const vector<vector<int>>& bow_matrix) {
    vector<vector<double>> tf_matrix = vector<vector<double>>(
        bow_matrix.size(),
        vector<double>(bow_matrix[0].size(), 0.0)
    );

    for (int i = 0; i < bow_matrix.size(); i++) {
        int total_words = 0;

        for (int count : bow_matrix[i]) {
            total_words += count;
        }

        if (total_words == 0) continue;

        for (int j = 0; j < bow_matrix[i].size(); j++) {
            tf_matrix[i][j] = static_cast<double>(bow_matrix[i][j]) / total_words;
        }
    }

    return tf_matrix;
}


inline vector<double> computeIDF(const vector<vector<int>>& bow_matrix) {
    int num_docs = (int)bow_matrix.size();
    int vocab_size = (int)bow_matrix[0].size();

    vector<double> idf(vocab_size, 0.0);

    for (int j = 0; j < vocab_size; j++) {
        int docs_containing_word = 0;
        for (int i = 0; i < num_docs; i++) {
            if (bow_matrix[i][j] > 0) docs_containing_word++;
        }

        // Smoothed IDF
        idf[j] = log(static_cast<double>(num_docs + 1) /
                     static_cast<double>(docs_containing_word + 1)) + 1.0;
    }

    return idf;
}


inline vector<vector<double>> computeTFIDF(
    const vector<vector<double>>& tf_matrix,
    const vector<double>& idf
) {
    vector<vector<double>> tfidf_matrix = tf_matrix;

    for (int i = 0; i < tf_matrix.size(); i++) {
        for (int j = 0; j < tf_matrix[i].size(); j++) {
            tfidf_matrix[i][j] = tf_matrix[i][j] * idf[j];
        }
    }

    return tfidf_matrix;
}
