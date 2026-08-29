// ContextMemory core — ETMC primitive types and data model.
//
// Event-Sourced Temporal Memory Compiler (ETMC):
// raw interaction is treated as an immutable event stream (Episode). The
// system compiles that stream into progressively more useful representations
// but never destroys the evidence required to recover or audit a decision:
//
//   Episode          immutable raw turn / tool trace (evidence, audit, recovery)
//   MemoryCell       compact self-contained fact with subject/predicate/object,
//                    event+system timestamps, version chain, tags
//   StateProjection  current answer keyed by (container, subject, predicate)
//
// Two clocks per cell:
//   observed_at  system time — when ContextMemory learned the statement
//   valid_from   event time  — when the statement became true in the world
// A cell is usable at time t only if it was observed by t AND its validity
// window covers t. This distinguishes "I moved last month" (said today) from
// "I lived in New York last year" and handles late-arriving corrections.

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cmcore {

// Unix epoch milliseconds. kNever = "no end".
using Timestamp = int64_t;

inline constexpr Timestamp kNever = std::numeric_limits<Timestamp>::max();
inline constexpr uint64_t kNoId = 0;

enum class CellKind : uint8_t {
    World = 0,       // objective facts about the external world
    Preference = 1,  // user preferences; strengthen with repetition
    Opinion = 2,     // subjective beliefs, carry a confidence
    Experience = 3,  // episodic / experience
    Procedure = 4,   // procedural knowledge ("how to")
};

enum class CellStatus : uint8_t {
    Active = 0,      // retrievable in current state
    Superseded = 1,  // replaced by a later version
    Expired = 2,     // validity window closed without replacement
    Forgotten = 3,   // explicitly removed from active retrieval
    Disputed = 4,    // conflicting evidence, held out pending adjudication
};

enum class EdgeType : uint8_t {
    Updates = 0,  // supersedes: old cell no longer current
    Extends = 1,  // adds detail, both remain valid
    Derives = 2,  // inferred claim from supporting cells
    Causal = 3,   // cause -> effect
    Related = 4,  // associative link
};

enum class TimeMode : uint8_t {
    Current = 0,     // "now", "currently" -> active projection, no old versions
    Historical = 1,  // "before", "used to" -> prior version at a time
    Interval = 2,    // explicit date range
    Relative = 3,    // "last week", "two months ago" -> resolved against now
    None = 4,        // no temporal constraint
};

enum class RelationMode : uint8_t {
    Direct = 0,    // single-hop recall
    MultiHop = 1,  // traverse edges between multiple entities
    Causal = 2,    // causal/procedural reasoning
    None = 3,
};

// --- immutable raw evidence -------------------------------------------------

struct Episode {
    uint64_t id = 0;
    uint64_t container = 0;
    std::string role;         // user | assistant | tool | system
    std::string content;      // raw text / tool trace
    Timestamp observed_at = 0;
    uint64_t session_id = 0;  // source session grouping
    uint64_t content_hash = 0;  // exact dedup
};

// --- retrieval unit ---------------------------------------------------------

struct MemoryCell {
    uint64_t id = 0;
    uint64_t container = 0;
    std::string subject;      // canonical entity id / name
    std::string predicate;    // normalized attribute or relation
    std::string object;       // normalized value
    std::string text;         // self-contained natural-language statement
    CellKind kind = CellKind::World;
    uint64_t source_episode = 0;  // immutable evidence
    uint32_t source_begin = 0;    // char offsets into source episode
    uint32_t source_end = 0;
    Timestamp observed_at = 0;    // system time
    Timestamp valid_from = 0;     // event time
    Timestamp valid_until = kNever;
    CellStatus status = CellStatus::Active;
    float confidence = 1.0f;
    float salience = 0.5f;
    uint32_t access_heat = 0;
    uint64_t root_id = 0;   // version-chain root
    uint64_t parent_id = 0; // prior version, if this is an update
    std::string source_ref; // external provenance reference
    std::vector<std::string> tags;
    std::vector<uint64_t> entity_ids;
    uint64_t content_hash = 0;

    // A cell is usable at time t when it was observed by t, not forgotten,
    // and its validity window covers t. Observed-time gating handles
    // late-arriving events: the agent cannot know tomorrow what arrives today.
    bool active_at(Timestamp t) const {
        if (status == CellStatus::Forgotten || status == CellStatus::Disputed)
            return false;
        if (observed_at > t) return false;
        if (valid_from > t) return false;
        if (valid_until <= t) return false;
        return true;
    }

    bool status_allows_history() const {
        return status == CellStatus::Superseded ||
               status == CellStatus::Expired;
    }
};

struct Edge {
    uint64_t id = 0;
    uint64_t container = 0;
    EdgeType type = EdgeType::Related;
    uint64_t from_id = 0;
    uint64_t to_id = 0;
    Timestamp created_at = 0;
    bool deleted = false;
};

struct Entity {
    uint64_t id = 0;
    uint64_t container = 0;
    std::string name;
    uint64_t fact_ref = 0;  // first cell that mentions it
};

// --- current-state projection -----------------------------------------------

struct StateProjection {
    uint64_t container = 0;
    std::string subject;
    std::string predicate;
    uint64_t active_cell = 0;   // the current winning cell
    uint64_t root_id = 0;       // version-chain root
    uint64_t version_count = 0;
    Timestamp updated_at = 0;
};

}  // namespace cmcore