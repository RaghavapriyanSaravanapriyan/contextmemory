// ContextMemory core — the ETMC store.
//
// Write path: capture immutable episodes (cheap, synchronous, no LLM), then
// reconcile compact memory cells produced by the extraction layer. Reconcile
// is deterministic first: exact-hash dedup, then subject/predicate versioning
// with validity windows, then semantic ambiguity resolution only when the
// caller supplies it. State projections maintain the current answer per
// (container, subject, predicate).
//
// Read path: compile a query into a bounded QueryPlan without an LLM, run the
// candidate channels (lexical, dense, entity, temporal, relation, projection),
// fuse with RRF, and pack the minimum-sufficient evidence under a token budget.
// Persistence: append-only binary journal replayed on load.

#pragma once

#include "index.hpp"
#include "types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cmcore {

// --- write ops --------------------------------------------------------------

// A compact cell produced by extraction. Reconcile uses subject/predicate to
// version deterministically before any semantic adjudication.
struct CellInput {
    std::string subject;
    std::string predicate;
    std::string object;
    std::string text;
    CellKind kind = CellKind::World;
    Timestamp observed_at = 0;  // system time; 0 = now
    Timestamp valid_from = 0;   // event time; 0 = observed_at
    float confidence = 1.0f;
    float salience = 0.5f;
    std::string source_ref;  // episode/session ref
    uint32_t source_begin = 0;
    uint32_t source_end = 0;
    std::vector<std::string> tags;
    std::vector<std::string> entities;
};

// --- query compilation ------------------------------------------------------

struct QueryPlan {
    TimeMode time_mode = TimeMode::None;
    Timestamp time_start = 0;  // resolved interval (ms)
    Timestamp time_end = kNever;
    std::vector<std::string> entity_seeds;
    RelationMode relation_mode = RelationMode::None;
    uint32_t kind_mask = 0xFFFFFFFFu;  // allowed CellKind bits
    std::vector<std::string> tags;
    uint32_t candidate_cap = 32;    // 4 | 8 | 16 | 32
    uint32_t expansion_cap = 1;     // graph hops
    size_t token_budget = 512;      // hard evidence cap
    bool fallback = true;           // widen one level when coverage is low
    std::string subject_hint;       // e.g. "user"
    std::string predicate_hint;     // e.g. "location" (direct projection hit)
    std::string text;               // original query (lexical channel)
};

// --- read results -----------------------------------------------------------

struct SearchResult {
    uint64_t cell_id = 0;
    std::string text;
    std::string subject;
    std::string predicate;
    std::string object;
    float score = 0.0f;
    CellKind kind = CellKind::World;
    CellStatus status = CellStatus::Active;
    float confidence = 1.0f;
    float salience = 0.5f;
    uint32_t access_heat = 0;
    Timestamp valid_from = 0;
    Timestamp valid_until = kNever;
    uint64_t root_id = 0;
    uint64_t parent_id = 0;
    std::string source_ref;
    std::vector<std::string> tags;
    bool projection_hit = false;  // selected from a state projection
};

struct EvidenceItem {
    SearchResult cell;
    bool covers_current = false;
    bool covers_historical = false;
    bool covers_relation = false;
};

struct EvidencePack {
    std::vector<EvidenceItem> items;
    size_t tokens = 0;
    size_t budget = 0;
    bool sufficient = false;  // enough evidence to answer, else abstain
    bool used_fallback = false;
};

struct SearchTrace {
    uint32_t candidates_seen = 0;
    uint32_t lexical_hits = 0;
    uint32_t dense_hits = 0;
    uint32_t entity_hits = 0;
    uint32_t projection_hits = 0;
    bool routed_by_projection = false;
    bool used_fallback = false;
    TimeMode time_mode = TimeMode::None;
    RelationMode relation_mode = RelationMode::None;
};

struct CompiledQuery {
    QueryPlan plan;
    SearchTrace trace;
};

struct ProfileResult {
    std::vector<SearchResult> static_facts;   // stable, high-salience
    std::vector<SearchResult> dynamic_facts;  // recent activity
};

class Store {
public:
    explicit Store(std::string container_tag);

    // --- capture (synchronous, no LLM) -------------------------------------
    uint64_t capture_episode(const Episode& ep);

    // --- reconcile (deterministic first) -----------------------------------
    // Returns the winning cell id (existing on dedup, new on create/update).
    // Sets projection when subject+predicate are present.
    uint64_t reconcile(const CellInput& in);

    // --- projections --------------------------------------------------------
    const StateProjection* projection(const std::string& subject,
                                      const std::string& predicate) const;
    void set_access_heat(uint64_t cell_id, uint32_t heat);
    void bump_access(uint64_t cell_id);

    // --- query compilation --------------------------------------------------
    CompiledQuery compile(const std::string& question, Timestamp at_time) const;

    // --- read path ----------------------------------------------------------
    std::vector<SearchResult> search(const QueryPlan& plan,
                                     std::span<const float> query_vec) const;
    EvidencePack pack(const std::vector<SearchResult>& ranked,
                      const QueryPlan& plan) const;

    ProfileResult profile(Timestamp at_time, uint32_t top_k) const;

    // --- low-level ops (tests / callers that bypass reconcile) -------------
    uint64_t create_cell(const MemoryCell& cell);
    void add_embedding(uint64_t cell_id, std::span<const float> vec);
    void link(EdgeType type, uint64_t from, uint64_t to, Timestamp at);

    // --- introspection ------------------------------------------------------
    const MemoryCell* cell(uint64_t id) const;
    const Episode* episode(uint64_t id) const;
    size_t cell_count() const { return cells_.size(); }
    size_t edge_count() const { return edges_.size(); }
    size_t entity_count() const { return entities_.size(); }
    size_t episode_count() const { return episodes_.size(); }
    size_t projection_count() const { return projections_.size(); }
    uint64_t container() const { return container_; }

    // --- persistence --------------------------------------------------------
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    uint64_t next_id();
    void add_edge(EdgeType type, uint64_t from, uint64_t to, Timestamp at);
    void index_cell(const MemoryCell& c);
    void unindex_cell(uint64_t cell_id);
    void update_projection(const MemoryCell& c);
    uint64_t ensure_entity(const std::string& name, uint64_t cell_id);
    uint64_t resolve_entity(const std::string& name) const;
    std::vector<uint64_t> active_candidates(const QueryPlan& plan,
                                            Timestamp at) const;
    void add_expanded(const std::vector<SearchResult>& hits,
                      const QueryPlan& plan, Timestamp at,
                      std::vector<SearchResult>& out) const;
    std::pair<std::string, std::string> infer_subject_predicate(
        const std::string& question,
        const std::vector<std::string>& entities) const;
    void resolve_relative_time(QueryPlan& plan, Timestamp at_time) const;

    std::string container_tag_;
    uint64_t container_;
    std::vector<MemoryCell> cells_;
    std::vector<Edge> edges_;
    std::vector<Entity> entities_;
    std::vector<Episode> episodes_;
    // key: subject + '\x1f' + predicate -> projection
    std::vector<StateProjection> projections_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> entity_to_cells_;
    std::unordered_map<std::string, std::vector<uint64_t>> tag_to_cells_;
    std::unordered_map<std::string, uint64_t> content_hash_to_cell_;
    uint64_t id_counter_ = 1;

    Bm25Index bm25_;
    VectorIndex vectors_;
};

}  // namespace cmcore