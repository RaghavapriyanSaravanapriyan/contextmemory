// ContextMemory core — index implementations (tokenizer, BM25, vectors).

#include "cmcore/index.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <unordered_set>

#if defined(__AVX2__) && !defined(CMCORE_NO_AVX)
#include <immintrin.h>
#endif

namespace cmcore {

namespace {

// Small English stopword list. Kept conservative: LongMemEval questions and
// extracted facts share functional words ("user", "preference") whose IDF is
// negligible, so aggressive removal is unnecessary.
const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "of", "at", "by",
        "for", "with", "from", "to", "in", "on", "is", "are", "was", "were",
        "be", "been", "being", "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "their", "our", "not", "no", "do",
        "does", "did", "have", "has", "had", "as", "so", "too", "will",
        "would", "can", "could", "should", "about", "into", "over", "there",
        "what", "which", "who", "when", "where", "how", "why",
        "user", "agent", "system", "assistant", "question", "answer",
        "please", "does", "the", "this", "that",
    };
    return words;
}

}  // namespace

bool is_stopword(std::string_view token) {
    return stopwords().find(std::string(token)) != stopwords().end();
}

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            cur.push_back(static_cast<char>(std::tolower(
                static_cast<unsigned char>(ch))));
        } else {
            if (cur.size() > 1 && !is_stopword(cur)) {
                out.push_back(cur);
            }
            cur.clear();
        }
    }
    if (cur.size() > 1 && !is_stopword(cur)) {
        out.push_back(cur);
    }
    return out;
}

// --- Bm25Index -------------------------------------------------------------

void Bm25Index::add(uint64_t fact_id, std::span<const std::string> tokens) {
    remove(fact_id);
    std::unordered_map<std::string, uint32_t> tf;
    for (const auto& tok : tokens) {
        tf[tok]++;
    }
    uint32_t len = 0;
    for (const auto& [term, count] : tf) {
        postings_[term].emplace_back(fact_id, count);
        len += count;
    }
    doc_len_[fact_id] = len;
    total_tokens_ += len;
}

void Bm25Index::remove(uint64_t fact_id) {
    auto it = doc_len_.find(fact_id);
    if (it == doc_len_.end()) return;
    for (auto& [term, postings] : postings_) {
        auto& vec = postings;
        vec.erase(
            std::remove_if(vec.begin(), vec.end(),
                           [fact_id](const auto& p) { return p.first == fact_id; }),
            vec.end());
    }
    total_tokens_ -= it->second;
    doc_len_.erase(it);
    // Drop terms with no postings to keep idf() cheap and correct.
    std::erase_if(postings_, [](const auto& kv) { return kv.second.empty(); });
}

void Bm25Index::update(uint64_t fact_id, std::span<const std::string> tokens) {
    add(fact_id, tokens);
}

std::vector<std::pair<uint64_t, float>> Bm25Index::score(
    std::span<const std::string> query_tokens,
    std::span<const uint64_t> candidates,
    size_t top_k) const {
    const size_t n = doc_len_.size();
    if (n == 0) return {};
    const double avg_dl =
        n == 0 ? 1.0 : static_cast<double>(total_tokens_) / n;

    std::unordered_map<uint64_t, float> scores;
    if (candidates.empty()) {
        scores.reserve(n * 2);
        for (const auto& [fid, len] : doc_len_) scores[fid] = 0.0f;
    } else {
        scores.reserve(candidates.size() * 2);
        for (uint64_t fid : candidates) {
            if (doc_len_.find(fid) != doc_len_.end()) scores[fid] = 0.0f;
        }
    }
    if (scores.empty()) return {};

    // Accumulate BM25 score per candidate.
    for (const auto& term : query_tokens) {
        auto pit = postings_.find(term);
        if (pit == postings_.end()) continue;
        const auto& postings = pit->second;
        const double df = static_cast<double>(postings.size());
        const double idf =
            std::log(1.0 + (static_cast<double>(n) - df + 0.5) / (df + 0.5));
        if (idf <= 0.0) continue;
        for (const auto& [fid, tf] : postings) {
            auto it = scores.find(fid);
            if (it == scores.end()) continue;
            auto lit = doc_len_.find(fid);
            if (lit == doc_len_.end()) continue;
            const double dl = static_cast<double>(lit->second);
            it->second += static_cast<float>(
                idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl)));
        }
    }

    std::vector<std::pair<uint64_t, float>> out(scores.begin(), scores.end());
    if (out.size() > top_k) {
        std::nth_element(out.begin(), out.begin() + top_k, out.end(),
                          [](const auto& a, const auto& b) {
                              if (a.second != b.second)
                                  return a.second > b.second;
                              return a.first < b.first;
                          });
        out.resize(top_k);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });
    return out;
}

// --- VectorIndex -----------------------------------------------------------

inline float dot_product(const float* a, const float* b, size_t n) {
#if defined(__AVX2__) && !defined(CMCORE_NO_AVX)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] +
                tmp[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

bool VectorIndex::has(uint64_t fact_id) const {
    return index_of_.count(fact_id) != 0;
}

void VectorIndex::add(uint64_t fact_id, std::span<const float> vec) {
    if (index_of_.count(fact_id)) remove(fact_id);
    if (vec.empty()) return;  // nothing to index
    if (dim_ == 0) {
        dim_ = vec.size();
    } else if (vec.size() != dim_) {
        // Dimension lock: a mismatched vector would corrupt every later
        // dot_product (heap OOB read of dim_ floats from a short vector).
        // Reject loudly-silent: keep the first dimension, drop the write.
        return;
    }
    // Normalize so cosine == dot product.
    std::vector<float> v(vec.begin(), vec.end());
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    if (norm > 0.0) {
        const double inv = 1.0 / std::sqrt(norm);
        for (auto& x : v) x = static_cast<float>(x * inv);
    }
    index_of_[fact_id] = static_cast<uint32_t>(ids_.size());
    ids_.push_back(fact_id);
    vecs_.push_back(std::move(v));
}

void VectorIndex::remove(uint64_t fact_id) {
    auto it = index_of_.find(fact_id);
    if (it == index_of_.end()) return;
    const uint32_t idx = it->second;
    const uint64_t last_id = ids_.back();
    ids_[idx] = last_id;
    vecs_[idx] = std::move(vecs_.back());
    index_of_[last_id] = idx;
    ids_.pop_back();
    vecs_.pop_back();
    index_of_.erase(it);
}

float VectorIndex::similarity(uint64_t fact_id,
                              std::span<const float> query) const {
    auto it = index_of_.find(fact_id);
    if (it == index_of_.end() || query.size() != dim_) return 0.0f;
    const auto& v = vecs_[it->second];
    if (v.size() != dim_) return 0.0f;  // corrupt entry: never OOB
    return dot_product(v.data(), query.data(), dim_);
}

std::vector<std::pair<uint64_t, float>> VectorIndex::top_k(
    std::span<const float> query,
    std::span<const uint64_t> candidates,
    size_t top_k) const {
    std::vector<std::pair<uint64_t, float>> out;
    if (dim_ == 0 || query.size() != dim_) return out;
    const auto consider = [&](uint64_t fid) {
        auto it = index_of_.find(fid);
        if (it == index_of_.end()) return;
        const auto& v = vecs_[it->second];
        if (v.size() != dim_) return;  // corrupt entry: never OOB
        out.emplace_back(fid, dot_product(v.data(), query.data(), dim_));
    };
    if (candidates.empty()) {
        for (uint64_t fid : ids_) consider(fid);
    } else {
        for (uint64_t fid : candidates) consider(fid);
    }
    if (out.size() > top_k) {
        std::nth_element(out.begin(), out.begin() + top_k, out.end(),
                          [](const auto& a, const auto& b) {
                              if (a.second != b.second)
                                  return a.second > b.second;
                              return a.first < b.first;
                          });
        out.resize(top_k);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });
    return out;
}

}  // namespace cmcore