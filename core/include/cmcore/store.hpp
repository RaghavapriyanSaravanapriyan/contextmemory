// ContextMemory core — the temporal graph store.
//
// The Store is the single write/read surface of the core engine. Write path:
// atomic batches of typed ops (create / update / link / expire / forget)
// produced by the extraction layer; the LLM never touches the core directly.
// Read path: deterministic hybrid retrieval (BM25 + vector + entity boost +
// recency) fused with RRF, constrained by a token budget. Persistence: an
// append-only binary journal replayed on load.

#pragma once

#include "index.hpp"
#include "types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cmcore {

// A single write operation. Ops are the language the extraction layer speaks;
// the Store applies them atomically and journals them.
struct Op {
    enum class Kind : uint8_t {
        CreateFact,  // text, kind, is_static, confidence, ts (created + valid)
        UpdateFact,  // fact_id, text, ts: supersede fact_id with new version
        Link,        // fact_id -> other_id, edge_type
        Expire,      // fact_id, ts: close validity window (no replacement)
        Forget,      // fact_id, ts: transactional removal
        SetConfidence,  // fact_id, value
    };

    Kind kind = Kind::CreateFact;
    uint64_t fact_id = 0;
    uint64_t other_id = 0;
    EdgeType edge_type = EdgeType::Related;
    FactKind fact_kind = FactKind::World;
    bool is_static = false;
    float confidence = 1.0f;
    Timestamp ts = 0;
    std::string text;
    std::string ref;  // source reference for the fact
    std::vector<std::string> entities;  // entity names for CreateFact
};

struct SearchResult {
    uint64_t fact_id = 0;
    std::string text;
    float score = 0.0f;
    FactKind kind = FactKind::World;
    bool is_static = false;
    float confidence = 1.0f;
    Timestamp valid_from = 0;
    Timestamp invalid_at = kNever;
    std::string source_ref;
    uint64_t root_id = 0;
};

struct ProfileResult {
    std::vector<SearchResult> static_facts;
    std::vector<SearchResult> dynamic_facts;
};

struct SearchOptions {
    Timestamp at_time = 0;          // facts must be valid at this time
    uint32_t top_k = 15;
    size_t token_budget = 700;      // 0 = unlimited
    bool include_expired = false;   // include transactionally-superseded facts
    uint32_t expand_depth = 0;      // graph expansion hops around hits
};

class Store {
public:
    explicit Store(std::string container_tag);

    // --- write path ---
    uint64_t create_fact(const Fact& fact);
    void apply(const Op& op);
    void apply_batch(const std::vector<Op>& ops);  // validated: all-or-nothing
    void add_fact_embedding(uint64_t fact_id, std::span<const float> vec);

    // Create a typed edge between two facts (updates/extends/derives/...).
    void link(EdgeType type, uint64_t from, uint64_t to, Timestamp at);

    // Version an existing fact: closes the old fact's validity window and
    // creates a child version linked by an Updates edge. Returns the new id.
    uint64_t update_fact(uint64_t fact_id, const std::string& text,
                         Timestamp ts,
                         const std::vector<std::string>& entities = {});

    // Create a fact, registering any entity names and linking them.
    uint64_t add_fact(const std::string& text, FactKind kind, bool is_static,
                      float confidence, Timestamp ts, const std::string& ref,
                      const std::vector<std::string>& entities = {});

    // --- read path ---
    std::vector<SearchResult> search(
        const std::string& text,
        std::span<const float> query_vec,          // empty = skip vector channel
        std::span<const std::string> query_entities,  // entity boost
        const SearchOptions& opts) const;

    ProfileResult profile(const SearchOptions& opts) const;

    // --- introspection ---
    const Fact* fact(uint64_t id) const;
    size_t fact_count() const { return facts_.size(); }
    size_t edge_count() const { return edges_.size(); }
    size_t entity_count() const { return entities_.size(); }
    uint64_t container() const { return container_; }

    // --- persistence ---
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    uint64_t next_id();
    void add_edge(EdgeType type, uint64_t from, uint64_t to, Timestamp at);
    uint64_t apply_update(uint64_t fact_id, const std::string& text, Timestamp ts,
                          const std::vector<std::string>& entities);
    void index_fact(const Fact& fact);
    void unindex_fact(uint64_t fact_id);
    void ensure_entity(const std::string& name, uint64_t fact_id);
    uint64_t resolve_entity(const std::string& name) const;
    std::vector<uint64_t> active_candidates(Timestamp at,
                                            bool include_expired) const;
    void add_expanded(const std::vector<SearchResult>& hits, Timestamp at,
                      uint32_t depth, std::vector<SearchResult>& out) const;

    std::string container_tag_;
    uint64_t container_;
    std::vector<Fact> facts_;
    std::vector<Edge> edges_;
    std::vector<Entity> entities_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> entity_to_facts_;
    uint64_t id_counter_ = 1;

    Bm25Index bm25_;
    VectorIndex vectors_;
};

}  // namespace cmcore