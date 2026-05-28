#pragma once

#include <cctype>
#include <stdexcept>
#include <vector>
#include <string>
#include <unordered_map>

namespace nn {

class Vocab {
private:
    std::unordered_map<std::string, size_t> tokenToId_;
    std::vector<std::string> idToToken_;

public:
    size_t add(const std::string& token) {
        auto it = tokenToId_.find(token);
        if (it != tokenToId_.end()) {
            return it->second;
        }

        size_t id = idToToken_.size();
        tokenToId_[token] = id;
        idToToken_.push_back(token);
        return id;
    }

    size_t id(const std::string& token) const {
        auto it = tokenToId_.find(token);
        if (it != tokenToId_.end()) {
            return it->second;
        }
        throw std::runtime_error("Unknown token: " + token);
    }

    const std::string& token(size_t id) const {
        if (id >= idToToken_.size()) {
            throw std::runtime_error("Token id out of range");
        }
        return idToToken_[id];
    }

    size_t size() const {
        return idToToken_.size();
    }
};


inline std::vector<std::string> simpleTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;

    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            // White space, push token
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}


inline Vocab buildVocab(const std::vector<std::string>& tokens) {
    Vocab vocab;
    for (const auto& token : tokens) {
        vocab.add(token);
    }
    return vocab;
}


inline std::vector<size_t> encodeTokens(
    const std::vector<std::string>& tokens,
    const Vocab& vocab
) {
    std::vector<size_t> ids;
    ids.resize(tokens.size());

    for (const auto& token : tokens) {
        ids.push_back(vocab.id(token));
    }

    return ids;
}


} // namespace nn
