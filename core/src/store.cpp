// ContextMemory core — ETMC store implementation.
//
// Capture -> reconcile -> project -> compile -> search -> pack.
// Reconcile is deterministic first: exact dedup, subject/predicate versioning
// with bi-temporal validity, projection maintenance. The read path is LLM-free
// and bounded: cheap intent routing compiles a QueryPlan, channels run over a
// narrowed candidate region, RRF fuses, and a token-budget packer selects the
// minimum sufficient evidence. Persistence is an append-only binary journal.

#include "cmcore/store.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <unordered_set>

namespace cmcore {

namespace {

// --- binary helpers --------------------------------------------------------

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 kJournalMagic = 0x434D4A4F;  // "CMJO"
constexpr u8 kRecEpisode = 0x01;
constexpr u8 kRecCell = 0x02;
constexpr u8 kRecProjection = 0x03;
constexpr u8 kRecEdge = 0x04;
constexpr u8 kRecEntity = 0x05;
constexpr u8 kRecEmbedding = 0x06;

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
    bool str_vec_(std::vector<std::string>& out) {
        u32 n;
        if (!u32_(n)) return false;
        out.clear();
        out.reserve(n);
        for (u32 i = 0; i < n; ++i) {
            std::string s;
            if (!str_(s)) return false;
            out.push_back(std::move(s));
        }
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

Timestamp now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
        .count();
}

std::string lower(const std::string& s) {
    std::string out = s;
    for (auto& c : out)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool contains_any(const std::string& lower_text,
                  std::initializer_list<const char*> words) {
    for (const char* w : words) {
        if (lower_text.find(w) != std::string::npos) return true;
    }
    return false;
}

// --- serialization ----------------------------------------------------------

std::string serialize_episode(const Episode& e) {
    std::string s;
    put_u64(s, e.id);
    put_u64(s, e.container);
    put_str(s, e.role);
    put_str(s, e.content);
    put_u64(s, static_cast<u64>(e.observed_at));
    put_u64(s, e.session_id);
    put_u64(s, e.content_hash);
    return s;
}

bool deserialize_episode(const u8* data, size_t len, Episode& e) {
    Reader r{data, data + len};
    if (!r.u64_(e.id) || !r.u64_(e.container)) return false;
    if (!r.str_(e.role) || !r.str_(e.content)) return false;
    u64 v;
    if (!r.u64_(v)) return false;
    e.observed_at = static_cast<Timestamp>(v);
    if (!r.u64_(e.session_id) || !r.u64_(e.content_hash)) return false;
    return true;
}

std::string serialize_cell(const MemoryCell& c) {
    std::string s;
    put_u64(s, c.id);
    put_u64(s, c.container);
    put_str(s, c.subject);
    put_str(s, c.predicate);
    put_str(s, c.object);
    put_str(s, c.text);
    put_u8(s, static_cast<u8>(c.kind));
    put_u64(s, c.source_episode);
    put_u32(s, c.source_begin);
    put_u32(s, c.source_end);
    put_u64(s, static_cast<u64>(c.observed_at));
    put_u64(s, static_cast<u64>(c.valid_from));
    put_u64(s, static_cast<u64>(c.valid_until));
    put_u8(s, static_cast<u8>(c.status));
    put_f32(s, c.confidence);
    put_f32(s, c.salience);
    put_u32(s, c.access_heat);
    put_u64(s, c.root_id);
    put_u64(s, c.parent_id);
    {
        put_u32(s, static_cast<u32>(c.tags.size()));
        for (const auto& t : c.tags) put_str(s, t);
    }
    put_str(s, c.source_ref);
    {
        put_u32(s, static_cast<u32>(c.entity_ids.size()));
        for (uint64_t e : c.entity_ids) put_u64(s, e);
    }
    put_u64(s, c.content_hash);
    return s;
}

bool deserialize_cell(const u8* data, size_t len, MemoryCell& c) {
    Reader r{data, data + len};
    if (!r.u64_(c.id) || !r.u64_(c.container)) return false;
    if (!r.str_(c.subject) || !r.str_(c.predicate) || !r.str_(c.object) ||
        !r.str_(c.text))
        return false;
    u8 k;
    if (!r.u8_(k)) return false;
    c.kind = static_cast<CellKind>(k);
    if (!r.u64_(c.source_episode)) return false;
    if (!r.u32_(c.source_begin) || !r.u32_(c.source_end)) return false;
    u64 v;
    if (!r.u64_(v)) return false;
    c.observed_at = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    c.valid_from = static_cast<Timestamp>(v);
    if (!r.u64_(v)) return false;
    c.valid_until = static_cast<Timestamp>(v);
    if (!r.u8_(k)) return false;
    c.status = static_cast<CellStatus>(k);
    if (!r.f32_(c.confidence) || !r.f32_(c.salience)) return false;
    if (!r.u32_(c.access_heat)) return false;
    if (!r.u64_(c.root_id) || !r.u64_(c.parent_id)) return false;
    if (!r.str_vec_(c.tags)) return false;
    if (!r.str_(c.source_ref)) return false;
    u32 n;
    if (!r.u32_(n)) return false;
    c.entity_ids.clear();
    c.entity_ids.reserve(n);
    for (u32 i = 0; i < n; ++i) {
        u64 eid;
        if (!r.u64_(eid)) return false;
        c.entity_ids.push_back(eid);
    }
    if (!r.u64_(c.content_hash)) return false;
    return true;
}

std::string serialize_projection(const StateProjection& p) {
    std::string s;
    put_u64(s, p.container);
    put_str(s, p.subject);
    put_str(s, p.predicate);
    put_u64(s, p.active_cell);
    put_u64(s, p.root_id);
    put_u64(s, p.version_count);
    put_u64(s, static_cast<u64>(p.updated_at));
    return s;
}

bool deserialize_projection(const u8* data, size_t len, StateProjection& p) {
    Reader r{data, data + len};
    if (!r.u64_(p.container)) return false;
    if (!r.str_(p.subject) || !r.str_(p.predicate)) return false;
    if (!r.u64_(p.active_cell) || !r.u64_(p.root_id) ||
        !r.u64_(p.version_count))
        return false;
    u64 v;
    if (!r.u64_(v)) return false;
    p.updated_at = static_cast<Timestamp>(v);
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
    put_u64(s, en.fact_ref);
    return s;
}

bool deserialize_entity(const u8* data, size_t len, Entity& en) {
    Reader r{data, data + len};
    if (!r.u64_(en.id) || !r.u64_(en.container)) return false;
    if (!r.str_(en.name)) return false;
    return r.u64_(en.fact_ref);
}

std::string serialize_embedding(u64 cell_id, std::span<const float> vec) {
    std::string s;
    put_u64(s, cell_id);
    put_u32(s, static_cast<u32>(vec.size()));
    for (float x : vec) put_f32(s, x);
    return s;
}

bool deserialize_embedding(const u8* data, size_t len, u64& cell_id,
                           std::vector<float>& vec) {
    Reader r{data, data + len};
    if (!r.u64_(cell_id)) return false;
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

std::unordered_map<uint64_t, float> rrf_fuse(
    const std::vector<std::vector<std::pair<uint64_t, float>>>& channels) {
    std::unordered_map<uint64_t, float> fused;
    for (const auto& channel : channels) {
        size_t rank = 0;
        for (const auto& [cid, score] : channel) {
            fused[cid] += 1.0f / (60.0f + static_cast<float>(rank));
            ++rank;
        }
    }
    return fused;
}

// --- date helpers -----------------------------------------------------------

const std::unordered_map<std::string, int>& month_map() {
    static const std::unordered_map<std::string, int> m = {
        {"january", 1},  {"february", 2}, {"march", 3},     {"april", 4},
        {"may", 5},      {"june", 6},     {"july", 7},      {"august", 8},
        {"september", 9},{"october", 10}, {"november", 11}, {"december", 12},
        {"jan", 1},      {"feb", 2},      {"mar", 3},       {"apr", 4},
        {"jun", 6},      {"jul", 7},      {"aug", 8},       {"sep", 9},
        {"sept", 9},     {"oct", 10},     {"nov", 11},      {"dec", 12},
    };
    return m;
}

Timestamp epoch_ms(int y, int mo, int d) {
    // Days from civil epoch (1970-01-01) — Howard Hinnant's algorithm.
    y -= mo <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153u * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const long long days = era * 146097LL + static_cast<long long>(doe) - 719468LL;
    return static_cast<Timestamp>(days) * 86'400'000LL;
}

// Extract an explicit date from lowercase question text. Returns true and sets
// out when a date is found (YYYY/MM/DD, YYYY-MM-DD, "march 5 2024", "in 2023").
bool extract_date(const std::string& lower, Timestamp& out) {
    static const std::regex slash_re(R"((\d{4})[/-](\d{1,2})[/-](\d{1,2}))");
    std::smatch m;
    if (std::regex_search(lower, m, slash_re)) {
        out = epoch_ms(std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]));
        return true;
    }
    // "<month> <day>, <year>" or "<month> <year>" or "<month> <day> <year>"
    for (const auto& [name, mo] : month_map()) {
        const size_t pos = lower.find(name);
        if (pos == std::string::npos) continue;
        const std::string rest = lower.substr(pos + name.size());
        int day = 0;
        int year = 0;
        std::regex_search(rest, m, std::regex(R"((\d{1,2})[,\s]*(\d{4}))"));
        if (m.size() >= 3) {
            day = std::stoi(m[1]);
            year = std::stoi(m[2]);
        } else {
            std::regex_search(rest, m, std::regex(R"(\d{4})"));
            if (m.size() >= 1) year = std::stoi(m[0]);
        }
        if (year == 0) year = 2000;
        if (day == 0) day = 1;
        out = epoch_ms(year, mo, day);
        return true;
    }
    std::regex_search(lower, m, std::regex(R"(\b(1[0-9]{3}|20[0-9]{2})\b)"));
    if (m.size() >= 1) {
        out = epoch_ms(std::stoi(m[0]), 1, 1);
        return true;
    }
    return false;
}

}  // namespace

// --- Store ------------------------------------------------------------------

Store::Store(std::string container_tag)
    : container_tag_(std::move(container_tag)),
      container_(fnv1a(container_tag_)) {}

uint64_t Store::next_id() { return id_counter_++; }

uint64_t Store::capture_episode(const Episode& ep) {
    Episode e = ep;
    if (e.container == 0) e.container = container_;
    if (e.observed_at == 0) e.observed_at = now_ms();
    e.content_hash = e.content_hash ? e.content_hash : fnv1a(e.content);
    e.id = next_id();
    episodes_.push_back(std::move(e));
    return episodes_.back().id;
}

uint64_t Store::ensure_entity(const std::string& name, uint64_t cell_id) {
    uint64_t eid = resolve_entity(name);
    if (eid == 0) {
        Entity en;
        en.id = next_id();
        en.container = container_;
        en.name = name;
        en.fact_ref = cell_id;
        entities_.push_back(std::move(en));
        eid = entities_.back().id;
    }
    auto& facts = entity_to_cells_[eid];
    if (std::find(facts.begin(), facts.end(), cell_id) == facts.end()) {
        facts.push_back(cell_id);
    }
    return eid;
}

uint64_t Store::resolve_entity(const std::string& name) const {
    const std::string lname = lower(name);
    for (const auto& e : entities_) {
        if (e.container == container_ && lower(e.name) == lname) return e.id;
    }
    return 0;
}

void Store::index_cell(const MemoryCell& c) {
    auto toks = tokenize(c.text);
    bm25_.add(c.id, toks);
    for (const auto& t : c.tags) {
        tag_to_cells_[lower(t)].push_back(c.id);
    }
    content_hash_to_cell_[c.text] = c.id;
}

void Store::unindex_cell(uint64_t cell_id) {
    bm25_.remove(cell_id);
    vectors_.remove(cell_id);
    const MemoryCell* c = cell(cell_id);
    if (c) {
        for (const auto& t : c->tags) {
            auto it = tag_to_cells_.find(lower(t));
            if (it != tag_to_cells_.end()) {
                auto& v = it->second;
                std::erase(v, cell_id);
                if (v.empty()) tag_to_cells_.erase(it);
            }
        }
        content_hash_to_cell_.erase(c->text);
    }
}

uint64_t Store::create_cell(const MemoryCell& cell) {
    MemoryCell c = cell;
    c.id = next_id();
    c.container = container_;
    if (c.content_hash == 0) c.content_hash = fnv1a(c.text);
    cells_.push_back(std::move(c));
    index_cell(cells_.back());
    return cells_.back().id;
}

void Store::update_projection(const MemoryCell& c) {
    if (c.subject.empty() || c.predicate.empty()) return;
    const std::string key = c.subject + "\x1f" + c.predicate;
    uint64_t version_count = 1;
    uint64_t root = c.root_id ? c.root_id : c.id;
    if (c.parent_id) {
        // count ancestors
        uint64_t cur = c.parent_id;
        while (cur) {
            ++version_count;
            const MemoryCell* p = cell(cur);
            if (!p) break;
            cur = p->parent_id;
        }
    }
    for (auto& p : projections_) {
        if (p.container == container_ && p.subject == c.subject &&
            p.predicate == c.predicate) {
            p.active_cell = c.id;
            p.root_id = root;
            p.version_count = version_count;
            p.updated_at = c.observed_at;
            return;
        }
    }
    StateProjection p;
    p.container = container_;
    p.subject = c.subject;
    p.predicate = c.predicate;
    p.active_cell = c.id;
    p.root_id = root;
    p.version_count = version_count;
    p.updated_at = c.observed_at;
    projections_.push_back(std::move(p));
}

const StateProjection* Store::projection(const std::string& subject,
                                         const std::string& predicate) const {
    for (const auto& p : projections_) {
        if (p.container == container_ && p.subject == subject &&
            p.predicate == predicate)
            return &p;
    }
    return nullptr;
}

uint64_t Store::reconcile(const CellInput& in) {
    if (in.text.empty()) return 0;
    const Timestamp observed = in.observed_at ? in.observed_at : now_ms();
    const Timestamp valid = in.valid_from ? in.valid_from : observed;

    // 1. exact dedup — same text already active: no-op.
    auto hit = content_hash_to_cell_.find(in.text);
    if (hit != content_hash_to_cell_.end()) {
        const MemoryCell* existing = cell(hit->second);
        if (existing && existing->status == CellStatus::Active) {
            return existing->id;
        }
    }

    MemoryCell nc;
    nc.text = in.text;
    nc.subject = in.subject;
    nc.predicate = in.predicate;
    nc.object = in.object;
    nc.kind = in.kind;
    nc.observed_at = observed;
    nc.valid_from = valid;
    nc.valid_until = kNever;
    nc.confidence = in.confidence;
    nc.salience = in.salience;
    nc.source_ref = in.source_ref;
    nc.source_begin = in.source_begin;
    nc.source_end = in.source_end;
    nc.content_hash = fnv1a(in.text);

    bool out_of_order = false;
    if (!nc.subject.empty() && !nc.predicate.empty()) {
        const StateProjection* proj = projection(nc.subject, nc.predicate);
        if (proj && proj->active_cell) {
            const MemoryCell* cur = cell(proj->active_cell);
            if (cur) {
                if (valid >= cur->valid_from) {
                    // versioning: supersede the current cell
                    MemoryCell* old = nullptr;
                    for (auto& c : cells_) {
                        if (c.id == cur->id) {
                            old = &c;
                            break;
                        }
                    }
                    if (old) {
                        old->valid_until = valid;
                        old->status = CellStatus::Superseded;
                        nc.parent_id = old->id;
                        nc.root_id = old->root_id ? old->root_id : old->id;
                    }
                } else {
                    out_of_order = true;  // earlier event: keep both, link
                }
            }
        }
    }

    uint64_t id = create_cell(nc);
    MemoryCell* created = nullptr;
    for (auto& c : cells_) {
        if (c.id == id) {
            created = &c;
            break;
        }
    }

    for (const auto& name : in.entities) {
        const uint64_t eid = ensure_entity(name, id);
        if (created && std::find(created->entity_ids.begin(),
                                 created->entity_ids.end(),
                                 eid) == created->entity_ids.end()) {
            created->entity_ids.push_back(eid);
        }
    }
    for (const auto& t : in.tags) {
        if (!t.empty()) created->tags.push_back(t);
    }

    if (nc.parent_id) {
        add_edge(EdgeType::Updates, nc.parent_id, id, observed);
    } else if (out_of_order) {
        const StateProjection* proj = projection(nc.subject, nc.predicate);
        if (proj && proj->active_cell) {
            add_edge(EdgeType::Related, proj->active_cell, id, observed);
        }
    }
    if (!nc.subject.empty() && !nc.predicate.empty()) {
        update_projection(*created);
    }
    return id;
}

void Store::set_access_heat(uint64_t cell_id, uint32_t heat) {
    for (auto& c : cells_) {
        if (c.id == cell_id) {
            c.access_heat = heat;
            return;
        }
    }
}

void Store::bump_access(uint64_t cell_id) {
    for (auto& c : cells_) {
        if (c.id == cell_id) {
            if (c.access_heat < 0xFFFFFFFEu) ++c.access_heat;
            return;
        }
    }
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

void Store::add_embedding(uint64_t cell_id, std::span<const float> vec) {
    if (!cell(cell_id)) return;
    vectors_.add(cell_id, vec);
}

const MemoryCell* Store::cell(uint64_t id) const {
    for (const auto& c : cells_) {
        if (c.id == id) return &c;
    }
    return nullptr;
}

const Episode* Store::episode(uint64_t id) const {
    for (const auto& e : episodes_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

// --- query compilation ------------------------------------------------------

void Store::resolve_relative_time(QueryPlan& plan, Timestamp at_time) const {
    if (plan.time_mode != TimeMode::Relative &&
        plan.time_mode != TimeMode::Historical)
        return;
    // already resolved to an explicit interval?
    if (plan.time_end != kNever || plan.time_start != 0) return;
    // Simplest robust rule: default historical window = past 90 days.
    plan.time_end = at_time;
    plan.time_start = at_time - 90LL * 86'400'000LL;
}

std::pair<std::string, std::string> Store::infer_subject_predicate(
    const std::string& question,
    const std::vector<std::string>& entities) const {
    const std::string lq = lower(question);
    std::string subject;
    if (contains_any(lq, {" i ", "i ", " my ", "my ", "i'm", "i've", "i am",
                          "me ", " user "})) {
        subject = "user";
    }
    // If an explicit entity is present, prefer it as the subject for
    // non-self-directed questions.
    if (subject.empty() && !entities.empty()) subject = entities[0];

    std::string predicate;
    struct Lex {
        const char* pred;
        std::initializer_list<const char*> words;
    };
    static const Lex lex[] = {
        {"location", {"where", "live", "city", "address", "moved", "move",
                      "reside", "hometown", "located"}},
        {"employer", {"work", "employer", "company", "job", "joined",
                      "employ", "office", "boss", "employed"}},
        {"name", {"name", "called", "named"}},
        {"preference", {"prefer", "favorite", "favourite", "like", "love",
                        "want", "likes", "prefers"}},
        {"hobby", {"hobby", "enjoy", "interest"}},
        {"pet", {"pet", "dog", "cat"}},
        {"plan", {"planning", "plan", "trip", "travel", "going"}},
    };
    for (const auto& entry : lex) {
        for (const char* w : entry.words) {
            if (lq.find(w) != std::string::npos) {
                predicate = entry.pred;
                break;
            }
        }
        if (!predicate.empty()) break;
    }
    return {subject, predicate};
}

CompiledQuery Store::compile(const std::string& question,
                             Timestamp at_time) const {
    const std::string lq = lower(question);
    CompiledQuery cq;
    QueryPlan& plan = cq.plan;
    SearchTrace& trace = cq.trace;
    plan.text = question;

    // 1. Time mode
    if (contains_any(lq, {"now", "currently", "right now", "at the moment",
                          "today", "these days", "lately", "current"})) {
        plan.time_mode = TimeMode::Current;
        plan.time_end = at_time;
    } else if (contains_any(lq, {"before", "used to", "previously", "prior",
                                 "earlier", "then", "formerly", "in the past",
                                 "old", "was"})) {
        plan.time_mode = TimeMode::Historical;
        plan.time_end = at_time;
    } else {
        Timestamp date = 0;
        if (extract_date(lq, date)) {
            plan.time_mode = TimeMode::Interval;
            plan.time_start = date;
            plan.time_end = date + 86'400'000LL;
        } else if (contains_any(lq, {"ago", "last week", "last month",
                                     "last year", "yesterday", "recent"})) {
            plan.time_mode = TimeMode::Relative;
            plan.time_end = at_time;
        } else {
            plan.time_mode = TimeMode::None;
            plan.time_end = at_time;
        }
    }
    resolve_relative_time(plan, at_time);
    trace.time_mode = plan.time_mode;

    // 2. Entity seeds from known entities present in the question. Generic
    // role words ("user", "agent", "system") carry no identity and must not
    // seed the entity channel — they are stopword-level noise.
    for (const auto& e : entities_) {
        if (e.container != container_) continue;
        if (plan.entity_seeds.size() >= 8) break;
        const std::string lname = lower(e.name);
        if (lname.size() < 3) continue;
        if (lname == "user" || lname == "agent" || lname == "system" ||
            lname == "assistant" || lname == "the" || lname == "a")
            continue;
        if (lq.find(lname) != std::string::npos) {
            plan.entity_seeds.push_back(e.name);
        }
    }

    // 3. Relation mode
    if (contains_any(lq, {"between", "both", "related to", "how are",
                          "compared", "together", "combined", "connect",
                          "connects", "connection", "link", "links",
                          "how do", "how does"})) {
        plan.relation_mode = RelationMode::MultiHop;
        plan.expansion_cap = 2;
    } else if (contains_any(lq, {"why", "because", "caused", "led to",
                                 "resulted", "how to", "steps", "process",
                                 "reason"})) {
        plan.relation_mode = RelationMode::Causal;
        plan.expansion_cap = 2;
    } else {
        plan.relation_mode = RelationMode::Direct;
    }
    trace.relation_mode = plan.relation_mode;

    // 4. Tags — tokens of the question that match known tags.
    const auto toks = tokenize(question);
    for (const auto& tok : toks) {
        auto it = tag_to_cells_.find(tok);
        if (it != tag_to_cells_.end()) {
            plan.tags.push_back(it->first);
            if (plan.tags.size() >= 4) break;
        }
    }

    // 5. Subject/predicate hint for direct projection hits.
    std::tie(plan.subject_hint, plan.predicate_hint) =
        infer_subject_predicate(question, plan.entity_seeds);
    // A projection-bearing hint with no explicit temporal framing is a
    // "current state" question ("Where does the user live?"), even without the
    // word "now".
    if (plan.time_mode == TimeMode::None &&
        !plan.subject_hint.empty() && !plan.predicate_hint.empty() &&
        projection(plan.subject_hint, plan.predicate_hint) != nullptr) {
        plan.time_mode = TimeMode::Current;
        plan.time_end = at_time;
        trace.time_mode = plan.time_mode;
    }
    trace.routed_by_projection =
        !plan.subject_hint.empty() && !plan.predicate_hint.empty() &&
        projection(plan.subject_hint, plan.predicate_hint) != nullptr;

    // 6. Complexity → budget + candidate cap.
    const size_t qlen = question.size();
    const size_t nent = plan.entity_seeds.size();
    const bool multi = plan.relation_mode != RelationMode::Direct;
    if (multi || qlen > 160 || nent >= 2) {
        plan.token_budget = 700;
        plan.candidate_cap = 32;
    } else if (qlen > 80 || nent == 1) {
        plan.token_budget = 512;
        plan.candidate_cap = 16;
    } else {
        plan.token_budget = 384;
        plan.candidate_cap = 8;
    }

    return cq;
}

// --- read path --------------------------------------------------------------

std::vector<uint64_t> Store::active_candidates(const QueryPlan& plan,
                                               Timestamp at) const {
    std::vector<uint64_t> out;
    out.reserve(cells_.size());
    const Timestamp t0 = plan.time_start;
    const Timestamp t1 = plan.time_end != kNever ? plan.time_end : at;
    for (const auto& c : cells_) {
        if (c.container != container_) continue;
        if (!(plan.kind_mask & (1u << static_cast<u32>(c.kind)))) continue;
        bool keep = false;
        switch (plan.time_mode) {
            case TimeMode::Current:
            case TimeMode::None:
                keep = c.active_at(t1);
                break;
            case TimeMode::Historical:
                // cells observed by t1 whose window existed at some point
                if (c.observed_at <= t1) {
                    if (c.status == CellStatus::Active) {
                        keep = c.valid_from <= t1;
                    } else if (c.status_allows_history()) {
                        keep = c.valid_until <= t1;  // already closed
                    }
                }
                break;
            case TimeMode::Interval:
            case TimeMode::Relative: {
                if (c.observed_at <= t1 && c.status != CellStatus::Forgotten &&
                    c.status != CellStatus::Disputed) {
                    keep = c.valid_from <= t1 && c.valid_until > t0;
                }
                break;
            }
        }
        if (!keep) continue;
        // tag narrowing: if the plan routed to tags, the cell must match one.
        if (!plan.tags.empty()) {
            bool tagged = false;
            for (const auto& t : c.tags) {
                if (std::find(plan.tags.begin(), plan.tags.end(), lower(t)) !=
                    plan.tags.end()) {
                    tagged = true;
                    break;
                }
            }
            if (!tagged) continue;
        }
        out.push_back(c.id);
    }
    // project-direct hits: ensure the active chain cells are always present
    // even if a tag filter would drop them.
    if (!plan.subject_hint.empty() && !plan.predicate_hint.empty()) {
        const StateProjection* proj = projection(plan.subject_hint,
                                                 plan.predicate_hint);
        if (proj && proj->active_cell) {
            uint64_t cur = proj->active_cell;
            while (cur) {
                if (std::find(out.begin(), out.end(), cur) == out.end()) {
                    out.push_back(cur);
                }
                const MemoryCell* c = cell(cur);
                if (!c || c->parent_id == 0) break;
                cur = c->parent_id;
            }
        }
    }
    if (plan.candidate_cap > 0 && out.size() > plan.candidate_cap * 4u) {
        out.resize(plan.candidate_cap * 4u);
    }
    return out;
}

void Store::add_expanded(const std::vector<SearchResult>& hits,
                         const QueryPlan& plan, Timestamp at,
                         std::vector<SearchResult>& out) const {
    if (plan.expansion_cap == 0) return;
    std::unordered_set<uint64_t> seen;
    for (const auto& h : out) seen.insert(h.cell_id);
    std::vector<uint64_t> frontier;
    for (const auto& h : hits) frontier.push_back(h.cell_id);

    for (uint32_t hop = 0; hop < plan.expansion_cap && !frontier.empty(); ++hop) {
        std::vector<uint64_t> next;
        for (uint64_t cid : frontier) {
            for (const auto& e : edges_) {
                if (e.container != container_ || e.deleted) continue;
                if (e.type == EdgeType::Related && plan.expansion_cap < 2)
                    continue;
                uint64_t other = 0;
                if (e.from_id == cid) other = e.to_id;
                else if (e.to_id == cid) other = e.from_id;
                else continue;
                if (seen.count(other)) continue;
                const MemoryCell* c = cell(other);
                const bool historical_ok =
                    plan.time_mode == TimeMode::Historical &&
                    c && c->status_allows_history();
                if (c && (c->active_at(at) || historical_ok)) {
                    seen.insert(other);
                    SearchResult r;
                    r.cell_id = c->id;
                    r.text = c->text;
                    r.subject = c->subject;
                    r.predicate = c->predicate;
                    r.object = c->object;
                    r.kind = c->kind;
                    r.status = c->status;
                    r.confidence = c->confidence;
                    r.salience = c->salience;
                    r.valid_from = c->valid_from;
                    r.valid_until = c->valid_until;
                    r.root_id = c->root_id;
                    r.parent_id = c->parent_id;
                    r.source_ref = c->source_ref;
                    r.tags = c->tags;
                    out.push_back(std::move(r));
                    next.push_back(other);
                }
            }
        }
        frontier = std::move(next);
    }
}

std::vector<SearchResult> Store::search(const QueryPlan& plan,
                                        std::span<const float> query_vec) const {
    std::vector<SearchResult> results;
    const Timestamp at = plan.time_end != kNever ? plan.time_end : now_ms();

    auto candidates = active_candidates(plan, at);
    if (candidates.empty() && !plan.fallback) return results;
    if (candidates.empty() && plan.fallback) {
        // widen: full container, active at `at`.
        QueryPlan wide = plan;
        wide.tags.clear();
        wide.time_mode = TimeMode::None;
        wide.time_start = 0;
        wide.time_end = kNever;
        candidates = active_candidates(wide, at);
    }
    if (candidates.empty()) return results;

    // Channel 1: lexical BM25 over the narrowed region.
    auto bm25_ranked = bm25_.score(tokenize(plan.text), candidates,
                                   plan.candidate_cap * 2);
    std::erase_if(bm25_ranked, [](const auto& p) { return p.second <= 0.0f; });

    // Channel 2: dense vector, only if the region is bounded.
    std::vector<std::pair<uint64_t, float>> vec_ranked;
    if (!query_vec.empty() && vectors_.dim() == query_vec.size() &&
        candidates.size() <= plan.candidate_cap * 2u) {
        vec_ranked = vectors_.top_k(query_vec, candidates, plan.candidate_cap * 2);
        std::erase_if(vec_ranked,
                      [](const auto& p) { return p.second <= 0.0f; });
    }

    // Channel 3: entity adjacency.
    std::vector<std::pair<uint64_t, float>> entity_ranked;
    if (!plan.entity_seeds.empty()) {
        std::unordered_set<uint64_t> cand(candidates.begin(), candidates.end());
        std::unordered_map<uint64_t, float> link_count;
        for (const auto& name : plan.entity_seeds) {
            const uint64_t eid = resolve_entity(name);
            if (eid == 0) continue;
            auto it = entity_to_cells_.find(eid);
            if (it == entity_to_cells_.end()) continue;
            for (uint64_t cid : it->second) {
                if (cand.count(cid)) link_count[cid] += 1.0f;
            }
        }
        for (const auto& [cid, c] : link_count) entity_ranked.emplace_back(cid, c);
        std::sort(entity_ranked.begin(), entity_ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    // Channel 4: state projection chain (direct hit), time-aware.
    std::vector<std::pair<uint64_t, float>> proj_ranked;
    if (!plan.subject_hint.empty() && !plan.predicate_hint.empty()) {
        const StateProjection* proj = projection(plan.subject_hint,
                                                 plan.predicate_hint);
        if (proj && proj->active_cell) {
            // Walk the chain from the active cell back to the root, keeping
            // only members the agent had observed by query time.
            std::vector<std::pair<uint64_t, float>> chain;
            uint64_t cur = proj->active_cell;
            while (cur) {
                const MemoryCell* c = cell(cur);
                if (!c) break;
                if (c->observed_at <= at) {
                    chain.emplace_back(c->id, 0.0f);
                }
                if (c->parent_id == 0) break;
                cur = c->parent_id;
            }
            if (plan.time_mode == TimeMode::Current && !chain.empty()) {
                // Only the newest observed member is the current truth.
                chain = {chain.front()};
            }
            if (plan.time_mode == TimeMode::Historical) {
                // Older versions are the historical truth: weight the oldest
                // observed member highest so "before" surfaces it.
                float w = 1.5f;
                for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                    proj_ranked.emplace_back(it->first, w);
                    w -= 0.15f;
                    if (w < 1.0f) w = 1.0f;
                }
            } else {
                float w = (plan.time_mode == TimeMode::Current) ? 2.0f : 1.0f;
                for (auto& [cid, _] : chain) {
                    proj_ranked.emplace_back(cid, w);
                    (void)_;
                }
            }
        }
    }

    std::vector<std::vector<std::pair<uint64_t, float>>> channels;
    channels.push_back(bm25_ranked);
    if (!vec_ranked.empty()) channels.push_back(vec_ranked);
    if (!entity_ranked.empty()) channels.push_back(entity_ranked);
    if (!proj_ranked.empty()) channels.push_back(proj_ranked);

    auto fused = rrf_fuse(channels);

    // Deterministic rerank.
    std::vector<std::pair<uint64_t, float>> scored;
    scored.reserve(fused.size());
    for (const auto& [cid, base] : fused) {
        const MemoryCell* c = cell(cid);
        if (!c) continue;
        float score = base;
        // projection boost (from channel already, add explicit tiebreak)
        bool is_proj = false;
        for (const auto& [pcid, w] : proj_ranked) {
            if (pcid == cid) {
                is_proj = true;
                score += w * 0.5f;
                break;
            }
        }
        // historical preference: superseded versions ranked by recency of close
        if (plan.time_mode == TimeMode::Historical &&
            c->status_allows_history()) {
            score += 0.4f;
        }
        // current preference: the active cell of the projection wins clearly
        if (plan.time_mode == TimeMode::Current && is_proj) {
            score += 0.8f;
        }
        // salience + heat as gentle tiebreakers, never truth
        score += c->salience * 0.05f + std::log1p(c->access_heat) * 0.02f;
        scored.emplace_back(cid, score);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [cid, score] : scored) {
        const MemoryCell* c = cell(cid);
        if (!c) continue;
        // Current mode must never surface a stale version — unless that cell
        // is the projection's truth-at-`at` (the newest member the agent had
        // observed by query time, which may still be a superseded cell when a
        // late-arriving event has not yet been observed).
        bool proj_hit = false;
        for (const auto& [pcid, w] : proj_ranked) {
            if (pcid == cid) {
                proj_hit = true;
                break;
            }
        }
        if (plan.time_mode == TimeMode::Current && !c->active_at(at) &&
            !proj_hit)
            continue;
        SearchResult r;
        r.cell_id = c->id;
        r.text = c->text;
        r.subject = c->subject;
        r.predicate = c->predicate;
        r.object = c->object;
        r.score = score;
        r.kind = c->kind;
        r.status = c->status;
        r.confidence = c->confidence;
        r.salience = c->salience;
        r.access_heat = c->access_heat;
        r.valid_from = c->valid_from;
        r.valid_until = c->valid_until;
        r.root_id = c->root_id;
        r.parent_id = c->parent_id;
        r.source_ref = c->source_ref;
        r.tags = c->tags;
        r.projection_hit = proj_hit;
        results.push_back(std::move(r));
        if (results.size() >= plan.candidate_cap) break;
    }

    add_expanded(results, plan, at, results);
    if (results.size() > plan.candidate_cap) results.resize(plan.candidate_cap);
    return results;
}

EvidencePack Store::pack(const std::vector<SearchResult>& ranked,
                         const QueryPlan& plan) const {
    EvidencePack out;
    out.budget = plan.token_budget;
    if (ranked.empty()) return out;

    const bool need_current = plan.time_mode == TimeMode::Current;
    const bool need_historical = plan.time_mode == TimeMode::Historical;
    const bool need_relation = plan.relation_mode != RelationMode::Direct;
    const bool need_interval = plan.time_mode == TimeMode::Interval ||
                               plan.time_mode == TimeMode::Relative;

    size_t budget_left = plan.token_budget;
    for (const auto& r : ranked) {
        if (out.items.size() >= plan.candidate_cap) break;
        const size_t tok = r.text.size() / 4 + 1 + 8;  // ~8 header tokens
        if (budget_left < tok) {
            if (out.items.empty()) {
                continue;  // always allow at least the single best cell
            }
            break;
        }
        EvidenceItem item;
        item.cell = r;
        item.covers_current = r.projection_hit && need_current;
        item.covers_historical =
            (r.status == CellStatus::Superseded ||
             r.status == CellStatus::Expired) &&
            need_historical;
        item.covers_relation = need_relation && out.items.size() < 2;
        out.items.push_back(std::move(item));
        budget_left -= tok;
        out.tokens += tok;
        if (budget_left == 0) break;
    }

    bool covered = false;
    if (!out.items.empty()) {
        if (need_current) {
            covered = std::any_of(out.items.begin(), out.items.end(),
                                  [](const auto& i) {
                                      return i.covers_current ||
                                             i.cell.projection_hit;
                                  });
        } else if (need_historical) {
            covered = std::any_of(out.items.begin(), out.items.end(),
                                  [](const auto& i) {
                                      return i.covers_historical;
                                  });
        } else if (need_interval) {
            covered = true;
        } else if (need_relation) {
            covered = out.items.size() >= 2;
        } else {
            covered = true;
        }
    }
    out.sufficient = covered;
    return out;
}

ProfileResult Store::profile(Timestamp at_time, uint32_t top_k) const {
    ProfileResult out;
    for (const auto& c : cells_) {
        if (c.container != container_ || !c.active_at(at_time)) continue;
        SearchResult r;
        r.cell_id = c.id;
        r.text = c.text;
        r.subject = c.subject;
        r.predicate = c.predicate;
        r.object = c.object;
        r.kind = c.kind;
        r.status = c.status;
        r.confidence = c.confidence;
        r.salience = c.salience;
        r.valid_from = c.valid_from;
        r.valid_until = c.valid_until;
        r.root_id = c.root_id;
        r.parent_id = c.parent_id;
        r.source_ref = c.source_ref;
        r.tags = c.tags;
        if (c.kind == CellKind::World || c.kind == CellKind::Preference) {
            r.score = c.confidence + c.salience;
            out.static_facts.push_back(std::move(r));
        } else {
            r.score = static_cast<float>(c.observed_at);
            out.dynamic_facts.push_back(std::move(r));
        }
    }
    const auto desc = [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
    };
    std::sort(out.static_facts.begin(), out.static_facts.end(), desc);
    std::sort(out.dynamic_facts.begin(), out.dynamic_facts.end(), desc);
    if (out.static_facts.size() > top_k) out.static_facts.resize(top_k);
    if (out.dynamic_facts.size() > top_k) out.dynamic_facts.resize(top_k);
    return out;
}

// --- persistence -----------------------------------------------------------

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

    for (const auto& e : episodes_) emit(kRecEpisode, serialize_episode(e));
    for (const auto& c : cells_) emit(kRecCell, serialize_cell(c));
    for (const auto& p : projections_)
        emit(kRecProjection, serialize_projection(p));
    for (const auto& e : edges_) emit(kRecEdge, serialize_edge(e));
    for (const auto& en : entities_) emit(kRecEntity, serialize_entity(en));
    for (uint64_t cid : vectors_.ids()) {
        const std::vector<float>* vec = vectors_.vector_of(cid);
        if (vec) emit(kRecEmbedding, serialize_embedding(cid, *vec));
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

    cells_.clear();
    edges_.clear();
    entities_.clear();
    episodes_.clear();
    projections_.clear();
    entity_to_cells_.clear();
    tag_to_cells_.clear();
    content_hash_to_cell_.clear();
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
        if (rec_type == kRecEpisode) {
            Episode e;
            if (!deserialize_episode(payload, len, e))
                throw std::runtime_error("bad episode record");
            if (e.id >= id_counter_) id_counter_ = e.id + 1;
            episodes_.push_back(std::move(e));
        } else if (rec_type == kRecCell) {
            MemoryCell c;
            if (!deserialize_cell(payload, len, c))
                throw std::runtime_error("bad cell record");
            if (c.id >= id_counter_) id_counter_ = c.id + 1;
            cells_.push_back(std::move(c));
        } else if (rec_type == kRecProjection) {
            StateProjection p;
            if (!deserialize_projection(payload, len, p))
                throw std::runtime_error("bad projection record");
            projections_.push_back(std::move(p));
        } else if (rec_type == kRecEdge) {
            Edge e;
            if (!deserialize_edge(payload, len, e))
                throw std::runtime_error("bad edge record");
            if (e.id >= id_counter_) id_counter_ = e.id + 1;
            edges_.push_back(std::move(e));
        } else if (rec_type == kRecEntity) {
            Entity en;
            if (!deserialize_entity(payload, len, en))
                throw std::runtime_error("bad entity record");
            if (en.id >= id_counter_) id_counter_ = en.id + 1;
            entities_.push_back(std::move(en));
        } else if (rec_type == kRecEmbedding) {
            u64 cell_id;
            std::vector<float> vec;
            if (!deserialize_embedding(payload, len, cell_id, vec))
                throw std::runtime_error("bad embedding record");
            vectors_.add(cell_id, vec);
        } else {
            throw std::runtime_error("unknown journal record type");
        }
        pos += 13 + len;
    }

    // Rebuild derived indexes.
    for (const auto& c : cells_) {
        if (c.container != container_) continue;
        index_cell(c);
        for (uint64_t eid : c.entity_ids) {
            entity_to_cells_[eid].push_back(c.id);
        }
    }
}

}  // namespace cmcore