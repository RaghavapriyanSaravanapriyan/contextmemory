// ContextMemory core — lexical and vector indexes.
//
// All indexes are in-process and dependency-free: BM25 over a hand-rolled
// inverted index with a small stopword list, and a dense vector index using
// brute-force cosine similarity with an AVX2 fast path. At single-user scale
// (thousands of facts) brute-force is microseconds; HNSW is a measured
// follow-up, not a default.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cmcore {

// Lowercases and splits on non-alphanumeric boundaries, dropping stopwords
// and single characters.
std::vector<std::string> tokenize(std::string_view text);

bool is_stopword(std::string_view token);

// Okapi BM25 index. add/remove/update maintain postings and document
// statistics; score() ranks a candidate set against a query.
class Bm25Index {
public:
    static constexpr double k1 = 1.5;
    static constexpr double b = 0.75;

    void add(uint64_t fact_id, std::span<const std::string> tokens);
    void remove(uint64_t fact_id);
    void update(uint64_t fact_id, std::span<const std::string> tokens);

    // Score candidate facts (empty candidate set = all indexed facts).
    // Returns (fact_id, score) pairs, highest score first, at most top_k.
    std::vector<std::pair<uint64_t, float>> score(
        std::span<const std::string> query_tokens,
        std::span<const uint64_t> candidates,
        size_t top_k) const;

    size_t size() const { return doc_len_.size(); }

private:
    struct Posting {
        uint32_t tf;
    };
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, uint32_t>>>
        postings_;  // term -> (fact_id, term freq)
    std::unordered_map<uint64_t, uint32_t> doc_len_;
    uint64_t total_tokens_ = 0;
};

// Dense vector index with normalized vectors and SIMD dot product.
class VectorIndex {
public:
    // Returns true if the fact already has an embedding (dimension lock).
    bool has(uint64_t fact_id) const;
    size_t dim() const { return dim_; }

    void add(uint64_t fact_id, std::span<const float> vec);
    void remove(uint64_t fact_id);

    // Cosine similarity of an embedded fact against a query vector.
    // Returns 0.0 if the fact has no embedding.
    float similarity(uint64_t fact_id, std::span<const float> query) const;

    // Top-k facts by cosine similarity among candidates.
    std::vector<std::pair<uint64_t, float>> top_k(
        std::span<const float> query,
        std::span<const uint64_t> candidates,
        size_t top_k) const;

    size_t size() const { return ids_.size(); }

    // Persistence access: embedded fact ids and raw vectors.
    const std::vector<uint64_t>& ids() const { return ids_; }
    const std::vector<float>* vector_of(uint64_t fact_id) const {
        auto it = index_of_.find(fact_id);
        if (it == index_of_.end()) return nullptr;
        return &vecs_[it->second];
    }

private:
    size_t dim_ = 0;
    std::vector<uint64_t> ids_;
    std::vector<std::vector<float>> vecs_;  // normalized
    std::unordered_map<uint64_t, uint32_t> index_of_;
};

}  // namespace cmcore