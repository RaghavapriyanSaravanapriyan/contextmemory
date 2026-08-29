// ContextMemory core — primitive types and data model.
//
// The core models a bi-temporal knowledge graph: facts carry both a validity
// window (when they were true) and a transactional timeline (when they were
// ingested / superseded). This mirrors the model proven by Zep/Graphiti
// (validity + transactional timelines) combined with Supermemory's edge
// semantics (updates / extends / derives) and Hindsight's fact-belief
// separation (world / opinion / preference / episode).

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cmcore {

// Unix epoch milliseconds. kNever = "no end".
using Timestamp = int64_t;

inline constexpr Timestamp kNever = std::numeric_limits<Timestamp>::max();

enum class FactKind : uint8_t {
    World = 0,       // objective facts about the external world
    Opinion = 1,     // subjective beliefs, carry a confidence
    Preference = 2,  // user preferences; strengthen with repetition
    Episode = 3,     // contextual / episodic; decay with time
};

enum class EdgeType : uint8_t {
    Updates = 0,  // supersedes: old fact no longer valid
    Extends = 1,  // adds detail, both remain valid
    Derives = 2,  // inferred connection
    Related = 3,  // generic association
    Causal = 4,   // causal link
};

struct Fact {
    uint64_t id = 0;
    uint64_t container = 0;  // hash of the container tag
    std::string text;
    FactKind kind = FactKind::World;
    bool is_static = false;   // durable, high priority (Supermemory isStatic)
    float confidence = 1.0f;  // opinion reinforcement (Hindsight)
    Timestamp valid_from = 0;          // t_valid: when the fact started being true
    Timestamp invalid_at = kNever;     // t_invalid: when it stopped being true
    Timestamp created_at = 0;          // t'created: ingestion time
    Timestamp expired_at = kNever;     // t'expired: transactional supersession
    Timestamp forget_after = kNever;   // automatic expiry (Supermemory forgetAfter)
    uint64_t parent_id = 0;            // version chain: superseded fact
    uint64_t root_id = 0;              // version chain: original fact
    bool is_latest = true;             // newest version in its chain
    std::vector<uint64_t> entity_ids;  // linked entities
    uint64_t source_id = 0;            // episode / document id
    std::string source_ref;            // external provenance reference

    // A fact is usable at time t if it has been ingested, has not been
    // transactionally superseded or auto-expired, and its validity window
    // covers t.
    bool active_at(Timestamp t) const {
        if (created_at > t) return false;
        if (expired_at <= t) return false;
        if (forget_after <= t) return false;
        if (valid_from > t) return false;
        if (invalid_at <= t) return false;
        return true;
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
};

}  // namespace cmcore