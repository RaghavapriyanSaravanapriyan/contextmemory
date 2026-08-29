// ContextMemory core — temporal graph store implementation.
//
// Write path: atomic, validated batch ops. Read path: time-aware candidate
// filtering, then hybrid fusion (BM25 + vector + entity boost + recency)
// via Reciprocal Rank Fusion, token-budget-constrained assembly, and optional
// graph expansion. Persistence: append-only binary journal replayed on load.

#include "cmcore/store.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_set>

namespace cmcore {

namespace {

// --- binary helpers --------------------------------------------------------

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 kJournalMagic = 0x434D4A4F;  // "CMJO"
constexpr u8 kRecFact = 0x01;
constexpr u8 kRecEdge = 0x02;
constexpr u8 kRecEntity = 0x03;
constexpr u8 kRecEmbedding = 0x04;

// CRC-32 (IEEE 802.3, polynomial 0xEDB88320).
class Crc32 {
public:
    static u32 of(const void* data, size_t len) {
        struct Table {
            u32 v[256];
        };
        static const Table table = [] {
            Table t;
            for (u32 i = 0; i < 256; ++i) {
                u32 c = i;
                for (int k = 0; k < 8; ++k) {
                    c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                }
                t.v[i] = c;
            }
            return t;
        }();
        const auto* p = static_cast<const u8*>(data);
        u32 crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < len; ++i) {
            crc = table.v[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFFu;
    }
};

void put_u64(std::string& s, u64 v) {
    s.append(reinterpret_cast<const char*>(&v), 8);
}
void put_u32(std::string& s, u32 v) {
    s.append(reinterpret_cast<const char*>(&v), 4);
}
void put_u8(std::string& s, u8 v) { s.push_back(static_cast<char>(v)); }
void put_f32(std::string& s, float v) {
    s.append(reinterpret_cast<const char*>(&v), 4);
}
void put_str(std::string& s, const std::string& str) {
    put_u32(s, static_cast<u32>(str.size()));
    s.append(str);
}

struct Reader {
    const u8* p;
    const u8* end;

    bool need(size_t n) const { return p + n <= end; }
    bool u8_(u8& out) {
        if (!need(1)) return false;
        out = *p++;
        return true;
    }
    bool u32_(u32& out) {
        if (!need(4)) return false;
        std::memcpy(&out, p, 4);
        p += 4;
        return true;
    }
    bool u64_(u64& out) {
        if (!need(8)) return false;
        std::memcpy(&out, p, 8);
        p += 8;
        return true;
    }
    bool f32_(float& out) {
        if (!need(4)) return false;
        std::memcpy(&out, p, 4);
        p += 4;
        return true;
    }
    bool str_(std::string& out) {
        u32 len;
        if (!u32_(len) || !need(len)) return false;
        out.assign(reinterpret_cast<const char*>(p), len);
        p += len;
        return true;
    }
};

// FNV-1a 64-bit.
u64 fnv1a(const std::string& s) {
    u64 h = 14695981039346656037ull;
    for (char c : s) {
        h ^= static_cast<u8>(c);
        h *= 1099511628211ull;
    }
    return h;
}

// --- serialization ---------------------------------------------------------

// save() persists a point-in-time snapshot of the store (facts, edges,
// entities, embeddings) rather than an op log. This keeps ID lineage intact
// (parent/root chains and link targets must survive a round-trip) at the cost
// of re-writing the whole journal on each save — fine at memory-layer scale.

std::string serialize_fact(const Fact& f) {
    std::string s;
    put_u64(s, f.id);
    put_u64(s, f.container);
    put_str(s, f.text);
    put_u8(s, static_cast<u8>(f.kind));
    put_u8(s, f.is_static ? 1 : 0);
    put_f32(s, f.confidence);
    put_u64(s, static_cast<u64>(f.valid_from));
    put_u64(s, static_cast<u64>(f.invalid_at));
    put_u64(s, static_cast<u64>(f.created_at));
    put_u64(s, static_cast<u64>(f.expired_at));
    put_u64(s, static_cast<u64>(f.forget_after));
    put_u64(s, f.parent_id);
    put_u64(s, f.root_id);
    put_u8(s, f.is_latest ? 1 : 0);
    put_u32(s, static_cast<u32>(f.entity_ids.size()));
    for (uint64_t eid : f.entity_ids) put_u64(s, eid);
    put_u64(s, f.source_id);
    put_str(s, f.source_ref);
    return s;
}

bool deserialize_fact(const u8* data, size_t len, Fact& f) {
    Reader r{data, data + len};
    if (!r.u64_(f.id) || !r.u64_(f.container)) return false;
    if (!r.str_(f.text)) return false;
    u8 kind;
    if (!r.u8_(kind)) return false;
    f.kind = static_cast<FactKind>(kind);
    u8 flag;
    if (!r.u8_(flag)) return false;
    f.is_static = flag != 0;
    if (!r.f32_(f.confidence)) return false;
    u64 v;
    if (!r.u64_(v)) return false;
    f.valid_from = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    f.invalid_at = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    f.created_at = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    f.expired_at = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    f.forget_after = static_cast<Timestamp>(v);
    if (!r.u64_(f.parent_id) || !r.u64_(f.root_id)) return false;
    if (!r.u8_(flag)) return false;
    f.is_latest = flag != 0;
    u32 n;
    if (!r.u32_(n)) return false;
    f.entity_ids.clear();
    f.entity_ids.reserve(n);
    for (u32 i = 0; i < n; ++i) {
        u64 eid;
        if (!r.u64_(eid)) return false;
        f.entity_ids.push_back(eid);
    }
    if (!r.u64_(f.source_id)) return false;
    if (!r.str_(f.source_ref)) return false;
    return true;
}

std::string serialize_edge(const Edge& e) {
    std::string s;
    put_u64(s, e.id);
    put_u64(s, e.container);
    put_u8(s, static_cast<u8>(e.type));
    put_u64(s, e.from_id);
    put_u64(s, e.to_id);
    put_u64(s, static_cast<u64>(e.created_at));
    put_u8(s, e.deleted ? 1 : 0);
    return s;
}

bool deserialize_edge(const u8* data, size_t len, Edge& e) {
    Reader r{data, data + len};
    if (!r.u64_(e.id) || !r.u64_(e.container)) return false;
    u8 t;
    if (!r.u8_(t)) return false;
    e.type = static_cast<EdgeType>(t);
    if (!r.u64_(e.from_id) || !r.u64_(e.to_id)) return false;
    u64 v;
    if (!r.u64_(v)) return false;
    e.created_at = static_cast<Timestamp>(v);
    u8 flag;
    if (!r.u8_(flag)) return false;
    e.deleted = flag != 0;
    return true;
}

std::string serialize_entity(const Entity& en) {
    std::string s;
    put_u64(s, en.id);
    put_u64(s, en.container);
    put_str(s, en.name);
    return s;
}

bool deserialize_entity(const u8* data, size_t len, Entity& en) {
    Reader r{data, data + len};
    if (!r.u64_(en.id) || !r.u64_(en.container)) return false;
    return r.str_(en.name);
}

std::string serialize_embedding(u64 fact_id, std::span<const float> vec) {
    std::string s;
    put_u64(s, fact_id);
    put_u32(s, static_cast<u32>(vec.size()));
    for (float x : vec) put_f32(s, x);
    return s;
}

bool deserialize_embedding(const u8* data, size_t len, u64& fact_id,
                           std::vector<float>& vec) {
    Reader r{data, data + len};
    if (!r.u64_(fact_id)) return false;
    u32 n;
    if (!r.u32_(n)) return false;
    vec.clear();
    vec.reserve(n);
    for (u32 i = 0; i < n; ++i) {
        float x;
        if (!r.f32_(x)) return false;
        vec.push_back(x);
    }
    return true;
}

// --- fusion -----------------------------------------------------------------

// Reciprocal Rank Fusion over ranked lists of fact ids.
std::unordered_map<uint64_t, float> rrf_fuse(
    const std::vector<std::vector<std::pair<uint64_t, float>>>& channels) {
    std::unordered_map<uint64_t, float> fused;
    for (const auto& channel : channels) {
        size_t rank = 0;
        for (const auto& [fid, score] : channel) {
            if (fused.count(fid) == 0) fused[fid] = 0.0f;
            fused[fid] += 1.0f / (60.0f + static_cast<float>(rank));
            ++rank;
        }
    }
    return fused;
}

double fact_age_days(Timestamp at, Timestamp created) {
    if (at <= created) return 0.0;
    return static_cast<double>(at - created) / 86'400'000.0;
}

}  // namespace

// --- Store -----------------------------------------------------------------

Store::Store(std::string container_tag)
    : container_tag_(std::move(container_tag)),
      container_(fnv1a(container_tag_)) {}

uint64_t Store::next_id() { return id_counter_++; }

void Store::ensure_entity(const std::string& name, uint64_t fact_id) {
    uint64_t eid = resolve_entity(name);
    if (eid == 0) {
        Entity e;
        e.id = next_id();
        e.container = container_;
        e.name = name;
        entities_.push_back(std::move(e));
        eid = entities_.back().id;
    }
    auto& facts = entity_to_facts_[eid];
    if (std::find(facts.begin(), facts.end(), fact_id) == facts.end()) {
        facts.push_back(fact_id);
    }
}

uint64_t Store::resolve_entity(const std::string& name) const {
    for (const auto& e : entities_) {
        if (e.container == container_ &&
            e.name.size() == name.size() &&
            std::equal(e.name.begin(), e.name.end(), name.begin(),
                       [](char a, char b) {
                           return std::tolower(a) == std::tolower(b);
                       })) {
            return e.id;
        }
    }
    return 0;
}

void Store::index_fact(const Fact& fact) {
    auto toks = tokenize(fact.text);
    bm25_.add(fact.id, toks);
}

void Store::unindex_fact(uint64_t fact_id) {
    bm25_.remove(fact_id);
    vectors_.remove(fact_id);
}

uint64_t Store::create_fact(const Fact& fact) {
    uint64_t id = next_id();
    facts_.push_back(fact);
    facts_.back().id = id;
    facts_.back().container = container_;
    for (uint64_t eid : facts_.back().entity_ids) {
        entity_to_facts_[eid].push_back(id);
    }
    index_fact(facts_.back());
    return id;
}

void Store::apply(const Op& op) {
    switch (op.kind) {
        case Op::Kind::CreateFact: {
            Fact f;
            f.text = op.text;
            f.kind = op.fact_kind;
            f.is_static = op.is_static;
            f.confidence = op.confidence;
            f.valid_from = op.ts;
            f.created_at = op.ts;
            f.source_ref = op.ref;
            uint64_t id = create_fact(f);
            for (const auto& name : op.entities) {
                ensure_entity(name, id);
                facts_.back().entity_ids.push_back(resolve_entity(name));
            }
            break;
        }
        case Op::Kind::UpdateFact:
            apply_update(op.fact_id, op.text, op.ts, op.entities);
            break;
        case Op::Kind::Link:
            add_edge(op.edge_type, op.fact_id, op.other_id, op.ts);
            break;
        case Op::Kind::Expire: {
            auto it = std::find_if(
                facts_.begin(), facts_.end(),
                [&](const Fact& f) { return f.id == op.fact_id; });
            if (it != facts_.end() && op.ts < it->invalid_at) {
                it->invalid_at = op.ts;
                it->is_latest = false;
            }
            break;
        }
        case Op::Kind::Forget: {
            auto it = std::find_if(
                facts_.begin(), facts_.end(),
                [&](const Fact& f) { return f.id == op.fact_id; });
            if (it != facts_.end() && op.ts < it->expired_at) {
                it->expired_at = op.ts;
                it->is_latest = false;
            }
            break;
        }
        case Op::Kind::SetConfidence: {
            auto it = std::find_if(
                facts_.begin(), facts_.end(),
                [&](const Fact& f) { return f.id == op.fact_id; });
            if (it != facts_.end()) it->confidence = op.confidence;
            break;
        }
    }
}

void Store::apply_batch(const std::vector<Op>& ops) {
    // Validate first so the batch is all-or-nothing: every referenced fact
    // must exist before any mutation happens.
    for (const auto& op : ops) {
        const bool needs_fact =
            op.kind == Op::Kind::UpdateFact || op.kind == Op::Kind::Link ||
            op.kind == Op::Kind::Expire || op.kind == Op::Kind::Forget ||
            op.kind == Op::Kind::SetConfidence;
        if (needs_fact && !fact(op.fact_id)) return;
        if (op.kind == Op::Kind::Link && !fact(op.other_id)) return;
    }
    for (const auto& op : ops) apply(op);
}

uint64_t Store::apply_update(uint64_t fact_id, const std::string& text,
                             Timestamp ts,
                             const std::vector<std::string>& entities) {
    auto it = std::find_if(facts_.begin(), facts_.end(),
                           [&](const Fact& f) { return f.id == fact_id; });
    if (it == facts_.end()) return 0;
    it->invalid_at = ts;
    it->is_latest = false;
    Fact nf;
    nf.text = text;
    nf.kind = it->kind;
    nf.is_static = it->is_static;
    nf.confidence = it->confidence;
    nf.valid_from = ts;
    nf.created_at = ts;
    nf.source_ref = it->source_ref;
    nf.parent_id = it->id;
    nf.root_id = it->root_id != 0 ? it->root_id : it->id;
    nf.entity_ids = it->entity_ids;
    uint64_t nid = create_fact(nf);
    add_edge(EdgeType::Updates, it->id, nid, ts);
    for (const auto& name : entities) {
        ensure_entity(name, nid);
        facts_.back().entity_ids.push_back(resolve_entity(name));
    }
    return nid;
}

uint64_t Store::update_fact(uint64_t fact_id, const std::string& text,
                            Timestamp ts,
                            const std::vector<std::string>& entities) {
    return apply_update(fact_id, text, ts, entities);
}

uint64_t Store::add_fact(const std::string& text, FactKind kind, bool is_static,
                         float confidence, Timestamp ts, const std::string& ref,
                         const std::vector<std::string>& entities) {
    Fact f;
    f.text = text;
    f.kind = kind;
    f.is_static = is_static;
    f.confidence = confidence;
    f.valid_from = ts;
    f.created_at = ts;
    f.source_ref = ref;
    uint64_t id = create_fact(f);
    for (const auto& name : entities) {
        ensure_entity(name, id);
        facts_.back().entity_ids.push_back(resolve_entity(name));
    }
    return id;
}

void Store::add_edge(EdgeType type, uint64_t from, uint64_t to, Timestamp at) {
    Edge e;
    e.id = next_id();
    e.container = container_;
    e.type = type;
    e.from_id = from;
    e.to_id = to;
    e.created_at = at;
    edges_.push_back(std::move(e));
}

void Store::link(EdgeType type, uint64_t from, uint64_t to, Timestamp at) {
    add_edge(type, from, to, at);
}

void Store::add_fact_embedding(uint64_t fact_id, std::span<const float> vec) {
    if (!fact(fact_id)) return;
    vectors_.add(fact_id, vec);
}

const Fact* Store::fact(uint64_t id) const {
    for (const auto& f : facts_) {
        if (f.id == id) return &f;
    }
    return nullptr;
}

std::vector<uint64_t> Store::active_candidates(Timestamp at,
                                               bool include_expired) const {
    std::vector<uint64_t> out;
    out.reserve(facts_.size());
    for (const auto& f : facts_) {
        if (f.container != container_) continue;
        if (include_expired) {
            if (f.created_at <= at && f.valid_from <= at) out.push_back(f.id);
        } else if (f.active_at(at)) {
            out.push_back(f.id);
        }
    }
    return out;
}

void Store::add_expanded(const std::vector<SearchResult>& hits, Timestamp at,
                         uint32_t depth,
                         std::vector<SearchResult>& out) const {
    if (depth == 0) return;
    std::unordered_set<uint64_t> seen;
    for (const auto& h : out) seen.insert(h.fact_id);
    std::vector<uint64_t> frontier;
    for (const auto& h : hits) frontier.push_back(h.fact_id);

    for (uint32_t hop = 0; hop < depth && !frontier.empty(); ++hop) {
        std::vector<uint64_t> next;
        for (uint64_t fid : frontier) {
            for (const auto& e : edges_) {
                if (e.container != container_ || e.deleted) continue;
                if (e.type == EdgeType::Related) continue;  // skip weak links
                uint64_t other = 0;
                if (e.from_id == fid) other = e.to_id;
                else if (e.to_id == fid) other = e.from_id;
                else continue;
                if (seen.count(other)) continue;
                const Fact* f = fact(other);
                if (f && f->active_at(at)) {
                    seen.insert(other);
                    SearchResult r;
                    r.fact_id = f->id;
                    r.text = f->text;
                    r.score = 0.0f;  // expansion: no similarity score
                    r.kind = f->kind;
                    r.is_static = f->is_static;
                    r.confidence = f->confidence;
                    r.valid_from = f->valid_from;
                    r.invalid_at = f->invalid_at;
                    r.source_ref = f->source_ref;
                    r.root_id = f->root_id;
                    out.push_back(std::move(r));
                    next.push_back(other);
                }
            }
        }
        frontier = std::move(next);
    }
}

std::vector<SearchResult> Store::search(
    const std::string& text, std::span<const float> query_vec,
    std::span<const std::string> query_entities,
    const SearchOptions& opts) const {
    std::vector<SearchResult> results;

    const auto candidates = active_candidates(opts.at_time, opts.include_expired);
    if (candidates.empty()) return results;

    // Channel 1: BM25 lexical.
    auto bm25_tokens = tokenize(text);
    auto bm25_ranked = bm25_.score(bm25_tokens, candidates, opts.top_k * 2);
    // Zero-score entries carry no signal and must not enter the fusion (a
    // candidate that matches nothing should not be returned as a hit).
    std::erase_if(bm25_ranked, [](const auto& p) { return p.second <= 0.0f; });

    // Channel 2: vector semantic.
    std::vector<std::pair<uint64_t, float>> vec_ranked;
    if (!query_vec.empty() && vectors_.dim() == query_vec.size()) {
        vec_ranked = vectors_.top_k(query_vec, candidates, opts.top_k * 2);
        std::erase_if(vec_ranked,
                      [](const auto& p) { return p.second <= 0.0f; });
    }

    // Channel 3: entity linking. Resolved query entities surface their linked
    // facts as a first-class channel, not merely a boost: entity recall must
    // work even when the lexical and vector channels miss.
    std::vector<std::pair<uint64_t, float>> entity_ranked;
    if (!query_entities.empty()) {
        std::unordered_set<uint64_t> cand(candidates.begin(), candidates.end());
        std::unordered_map<uint64_t, float> link_count;
        for (const auto& name : query_entities) {
            uint64_t eid = resolve_entity(name);
            if (eid == 0) continue;
            auto it = entity_to_facts_.find(eid);
            if (it == entity_to_facts_.end()) continue;
            for (uint64_t fid : it->second) {
                if (cand.count(fid)) link_count[fid] += 1.0f;
            }
        }
        for (const auto& [fid, c] : link_count) {
            entity_ranked.emplace_back(fid, c);
        }
        std::sort(entity_ranked.begin(), entity_ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    std::vector<std::vector<std::pair<uint64_t, float>>> channels;
    channels.push_back(bm25_ranked);
    if (!vec_ranked.empty()) channels.push_back(vec_ranked);
    if (!entity_ranked.empty()) channels.push_back(entity_ranked);

    auto fused = rrf_fuse(channels);

    std::vector<std::pair<uint64_t, float>> scored;
    scored.reserve(fused.size());
    for (const auto& [fid, base] : fused) {
        float score = base;
        const Fact* f = fact(fid);
        if (f) {
            // Static facts are durable: no recency decay. Dynamic facts decay
            // with age (30-day half-life).
            if (!f->is_static) {
                const double age = fact_age_days(opts.at_time, f->created_at);
                score *= static_cast<float>(1.0 / (1.0 + age / 30.0));
            }
        }
        scored.emplace_back(fid, score);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    size_t budget_tokens = 0;
    for (const auto& [fid, score] : scored) {
        const Fact* f = fact(fid);
        if (!f) continue;
        SearchResult r;
        r.fact_id = f->id;
        r.text = f->text;
        r.score = score;
        r.kind = f->kind;
        r.is_static = f->is_static;
        r.confidence = f->confidence;
        r.valid_from = f->valid_from;
        r.invalid_at = f->invalid_at;
        r.source_ref = f->source_ref;
        r.root_id = f->root_id;
        results.push_back(std::move(r));
        if (opts.token_budget > 0) {
            budget_tokens += f->text.size() / 4 + 1;
            if (budget_tokens >= opts.token_budget && results.size() >= 2) {
                break;
            }
        }
        if (results.size() >= opts.top_k) break;
    }

    add_expanded(results, opts.at_time, opts.expand_depth, results);
    if (results.size() > opts.top_k) results.resize(opts.top_k);
    return results;
}

ProfileResult Store::profile(const SearchOptions& opts) const {
    ProfileResult out;
    const auto candidates = active_candidates(opts.at_time, opts.include_expired);
    for (uint64_t fid : candidates) {
        const Fact* f = fact(fid);
        if (!f) continue;
        SearchResult r;
        r.fact_id = f->id;
        r.text = f->text;
        r.kind = f->kind;
        r.is_static = f->is_static;
        r.confidence = f->confidence;
        r.valid_from = f->valid_from;
        r.invalid_at = f->invalid_at;
        r.source_ref = f->source_ref;
        r.root_id = f->root_id;
        if (f->is_static) {
            r.score = f->confidence;
            out.static_facts.push_back(std::move(r));
        } else {
            // Dynamic facts ranked by recency.
            r.score = static_cast<float>(f->created_at);
            out.dynamic_facts.push_back(std::move(r));
        }
    }
    auto by_score_desc = [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
    };
    std::sort(out.static_facts.begin(), out.static_facts.end(), by_score_desc);
    std::sort(out.dynamic_facts.begin(), out.dynamic_facts.end(),
              by_score_desc);
    if (out.static_facts.size() > opts.top_k) {
        out.static_facts.resize(opts.top_k);
    }
    if (out.dynamic_facts.size() > opts.top_k) {
        out.dynamic_facts.resize(opts.top_k);
    }
    return out;
}

// --- persistence -----------------------------------------------------------

// Snapshot journal: one length+CRC framed record per fact / edge / entity /
// embedding, preserving original IDs so parent/root chains and link targets
// survive a load. Replayed in order on load().

void Store::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open journal for write: " + path);

    const auto emit = [&](u8 rec_type, const std::string& payload) {
        std::string header;
        put_u32(header, kJournalMagic);
        put_u32(header, static_cast<u32>(payload.size()));
        const u32 crc = Crc32::of(payload.data(), payload.size());
        put_u32(header, crc);
        header.push_back(static_cast<char>(rec_type));
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    };

    for (const auto& f : facts_) emit(kRecFact, serialize_fact(f));
    for (const auto& e : edges_) emit(kRecEdge, serialize_edge(e));
    for (const auto& en : entities_) emit(kRecEntity, serialize_entity(en));
    for (uint64_t fid : vectors_.ids()) {
        const std::vector<float>* vec = vectors_.vector_of(fid);
        if (vec) emit(kRecEmbedding, serialize_embedding(fid, *vec));
    }
    out.flush();
    if (!out) throw std::runtime_error("failed writing journal: " + path);
}

void Store::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open journal for read: " + path);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<u8> buf(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(buf.data()), size);
    if (!in) throw std::runtime_error("failed reading journal: " + path);

    // Clear current state; the journal is authoritative on load.
    facts_.clear();
    edges_.clear();
    entities_.clear();
    entity_to_facts_.clear();
    id_counter_ = 1;

    size_t pos = 0;
    while (pos + 13 <= buf.size()) {
        const u8* rec = buf.data() + pos;
        u32 magic, len, crc;
        u8 rec_type;
        std::memcpy(&magic, rec, 4);
        std::memcpy(&len, rec + 4, 4);
        std::memcpy(&crc, rec + 8, 4);
        rec_type = rec[12];
        if (magic != kJournalMagic || pos + 13 + len > buf.size()) {
            throw std::runtime_error("corrupt journal record");
        }
        const u8* payload = rec + 13;
        if (Crc32::of(payload, len) != crc) {
            throw std::runtime_error("journal checksum mismatch");
        }
        if (rec_type == kRecFact) {
            Fact f;
            if (!deserialize_fact(payload, len, f)) {
                throw std::runtime_error("bad fact record");
            }
            if (f.id >= id_counter_) id_counter_ = f.id + 1;
            facts_.push_back(std::move(f));
        } else if (rec_type == kRecEdge) {
            Edge e;
            if (!deserialize_edge(payload, len, e)) {
                throw std::runtime_error("bad edge record");
            }
            if (e.id >= id_counter_) id_counter_ = e.id + 1;
            edges_.push_back(std::move(e));
        } else if (rec_type == kRecEntity) {
            Entity en;
            if (!deserialize_entity(payload, len, en)) {
                throw std::runtime_error("bad entity record");
            }
            if (en.id >= id_counter_) id_counter_ = en.id + 1;
            entities_.push_back(std::move(en));
        } else if (rec_type == kRecEmbedding) {
            u64 fact_id;
            std::vector<float> vec;
            if (!deserialize_embedding(payload, len, fact_id, vec)) {
                throw std::runtime_error("bad embedding record");
            }
            vectors_.add(fact_id, vec);
        } else {
            throw std::runtime_error("unknown journal record type");
        }
        pos += 13 + len;
    }

    // Rebuild derived structures from the restored facts.
    for (const auto& f : facts_) {
        if (f.container == container_) index_fact(f);
        for (uint64_t eid : f.entity_ids) {
            entity_to_facts_[eid].push_back(f.id);
        }
    }
}

}  // namespace cmcore