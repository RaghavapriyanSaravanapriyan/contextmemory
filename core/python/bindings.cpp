// ContextMemory core — nanobind Python bindings.
//
// Bound surface: Store with capture (episodes), reconcile (cells), state
// projections, query compilation, deterministic search, token-budget evidence
// packing, profiles, and snapshot persistence. The LLM orchestration never
// touches the core directly; it produces CellInput and consumes SearchResult /
// EvidencePack.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cmcore/store.hpp"

namespace nb = nanobind;
using namespace cmcore;

namespace {

nb::dict result_dict(const SearchResult& r) {
    nb::dict d;
    d["cell_id"] = r.cell_id;
    d["text"] = r.text;
    d["subject"] = r.subject;
    d["predicate"] = r.predicate;
    d["object"] = r.object;
    d["score"] = r.score;
    d["kind"] = static_cast<int>(r.kind);
    d["status"] = static_cast<int>(r.status);
    d["confidence"] = r.confidence;
    d["salience"] = r.salience;
    d["access_heat"] = r.access_heat;
    d["valid_from"] = r.valid_from;
    d["valid_until"] = r.valid_until;
    d["root_id"] = r.root_id;
    d["parent_id"] = r.parent_id;
    d["source_ref"] = r.source_ref;
    d["tags"] = r.tags;
    d["projection_hit"] = r.projection_hit;
    return d;
}

nb::list results_list(const std::vector<SearchResult>& results) {
    nb::list out;
    for (const auto& r : results) out.append(result_dict(r));
    return out;
}

nb::dict compiled_dict(const CompiledQuery& cq) {
    nb::dict d;
    d["time_mode"] = static_cast<int>(cq.plan.time_mode);
    d["time_start"] = cq.plan.time_start;
    d["time_end"] = cq.plan.time_end;
    d["entity_seeds"] = cq.plan.entity_seeds;
    d["relation_mode"] = static_cast<int>(cq.plan.relation_mode);
    d["kind_mask"] = cq.plan.kind_mask;
    d["tags"] = cq.plan.tags;
    d["candidate_cap"] = cq.plan.candidate_cap;
    d["expansion_cap"] = cq.plan.expansion_cap;
    d["token_budget"] = cq.plan.token_budget;
    d["fallback"] = cq.plan.fallback;
    d["subject_hint"] = cq.plan.subject_hint;
    d["predicate_hint"] = cq.plan.predicate_hint;
    d["text"] = cq.plan.text;
    nb::dict tr;
    tr["candidates_seen"] = cq.trace.candidates_seen;
    tr["lexical_hits"] = cq.trace.lexical_hits;
    tr["dense_hits"] = cq.trace.dense_hits;
    tr["entity_hits"] = cq.trace.entity_hits;
    tr["projection_hits"] = cq.trace.projection_hits;
    tr["routed_by_projection"] = cq.trace.routed_by_projection;
    tr["used_fallback"] = cq.trace.used_fallback;
    tr["time_mode"] = static_cast<int>(cq.trace.time_mode);
    tr["relation_mode"] = static_cast<int>(cq.trace.relation_mode);
    d["trace"] = tr;
    return d;
}

nb::dict pack_dict(const EvidencePack& p) {
    nb::dict d;
    nb::list items;
    for (const auto& item : p.items) {
        nb::dict id;
        id["cell"] = result_dict(item.cell);
        id["covers_current"] = item.covers_current;
        id["covers_historical"] = item.covers_historical;
        id["covers_relation"] = item.covers_relation;
        items.append(id);
    }
    d["items"] = items;
    d["tokens"] = p.tokens;
    d["budget"] = p.budget;
    d["sufficient"] = p.sufficient;
    d["used_fallback"] = p.used_fallback;
    return d;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "ContextMemory core engine (C++)";

    nb::class_<Store>(m, "Store")
        .def(nb::init<std::string>(), nb::arg("container_tag") = std::string(),
             "Create a store scoped to a container tag.")

        // --- capture ---
        .def("capture_episode",
             [](Store& s, const std::string& role, const std::string& content,
                int64_t observed_at, uint64_t session_id) {
                 Episode e;
                 e.role = role;
                 e.content = content;
                 e.observed_at = observed_at;
                 e.session_id = session_id;
                 return s.capture_episode(e);
             },
             nb::arg("role"), nb::arg("content"),
             nb::arg("observed_at") = 0, nb::arg("session_id") = 0,
             "Append an immutable episode; returns its id.")

        // --- reconcile ---
        .def("reconcile",
             [](Store& s, const std::string& subject,
                const std::string& predicate, const std::string& object,
                const std::string& text, int kind, int64_t observed_at,
                int64_t valid_from, float confidence, float salience,
                const std::string& source_ref, uint32_t source_begin,
                uint32_t source_end, const std::vector<std::string>& tags,
                const std::vector<std::string>& entities) {
                 CellInput in;
                 in.subject = subject;
                 in.predicate = predicate;
                 in.object = object;
                 in.text = text;
                 in.kind = static_cast<CellKind>(kind);
                 in.observed_at = observed_at;
                 in.valid_from = valid_from;
                 in.confidence = confidence;
                 in.salience = salience;
                 in.source_ref = source_ref;
                 in.source_begin = source_begin;
                 in.source_end = source_end;
                 in.tags = tags;
                 in.entities = entities;
                 return s.reconcile(in);
             },
             nb::arg("subject") = std::string(),
             nb::arg("predicate") = std::string(),
             nb::arg("object") = std::string(), nb::arg("text"),
             nb::arg("kind") = 0, nb::arg("observed_at") = 0,
             nb::arg("valid_from") = 0, nb::arg("confidence") = 1.0f,
             nb::arg("salience") = 0.5f,
             nb::arg("source_ref") = std::string(),
             nb::arg("source_begin") = 0u, nb::arg("source_end") = 0u,
             nb::arg("tags") = std::vector<std::string>(),
             nb::arg("entities") = std::vector<std::string>(),
             "Reconcile a cell (dedup / version / project); returns cell id.")

        // --- projections ---
        .def("projection",
             [](const Store& s, const std::string& subject,
                const std::string& predicate) -> nb::object {
                 const StateProjection* p = s.projection(subject, predicate);
                 if (!p) return nb::none();
                 nb::dict d;
                 d["subject"] = p->subject;
                 d["predicate"] = p->predicate;
                 d["active_cell"] = p->active_cell;
                 d["root_id"] = p->root_id;
                 d["version_count"] = p->version_count;
                 d["updated_at"] = p->updated_at;
                 return d;
             },
             nb::arg("subject"), nb::arg("predicate"),
             "Current state projection, or None.")
        .def("bump_access", &Store::bump_access, nb::arg("cell_id"))

        // --- query compilation ---
        .def("compile",
             [](const Store& s, const std::string& question, int64_t at_time) {
                 return compiled_dict(s.compile(question, at_time));
             },
             nb::arg("question"), nb::arg("at_time") = 0,
             "Compile a query into a bounded plan + trace.")

        // --- read path ---
        .def("search",
             [](const Store& s, const std::string& text, int64_t at_time,
                int time_mode, int64_t time_start, int64_t time_end,
                const std::vector<std::string>& entity_seeds, int relation_mode,
                uint32_t kind_mask, const std::vector<std::string>& tags,
                uint32_t candidate_cap, uint32_t expansion_cap,
                size_t token_budget, bool fallback,
                const std::string& subject_hint,
                const std::string& predicate_hint,
                const std::vector<float>& query_vec) {
                 QueryPlan plan;
                 plan.text = text;
                 plan.time_mode = static_cast<TimeMode>(time_mode);
                 plan.time_start = time_start;
                 plan.time_end = time_end;
                 plan.entity_seeds = entity_seeds;
                 plan.relation_mode = static_cast<RelationMode>(relation_mode);
                 plan.kind_mask = kind_mask;
                 plan.tags = tags;
                 plan.candidate_cap = candidate_cap;
                 plan.expansion_cap = expansion_cap;
                 plan.token_budget = token_budget;
                 plan.fallback = fallback;
                 plan.subject_hint = subject_hint;
                 plan.predicate_hint = predicate_hint;
                 return results_list(s.search(plan, query_vec));
             },
             nb::arg("text"), nb::arg("at_time") = 0,
             nb::arg("time_mode") = 0, nb::arg("time_start") = 0,
             nb::arg("time_end") = 0,
             nb::arg("entity_seeds") = std::vector<std::string>(),
             nb::arg("relation_mode") = 0, nb::arg("kind_mask") = 0xFFFFFFFFu,
             nb::arg("tags") = std::vector<std::string>(),
             nb::arg("candidate_cap") = 32u, nb::arg("expansion_cap") = 1u,
             nb::arg("token_budget") = size_t{512},
             nb::arg("fallback") = true,
             nb::arg("subject_hint") = std::string(),
             nb::arg("predicate_hint") = std::string(),
             nb::arg("query_vec") = std::vector<float>(),
             "Deterministic search over a compiled plan.")
        .def("pack",
             [](const Store& s, const std::string& text, int64_t at_time,
                int time_mode, int64_t time_start, int64_t time_end,
                uint32_t candidate_cap, uint32_t expansion_cap,
                size_t token_budget, int relation_mode) {
                 QueryPlan plan;
                 plan.text = text;
                 plan.time_mode = static_cast<TimeMode>(time_mode);
                 plan.time_start = time_start;
                 plan.time_end = time_end;
                 plan.candidate_cap = candidate_cap;
                 plan.expansion_cap = expansion_cap;
                 plan.token_budget = token_budget;
                 plan.relation_mode = static_cast<RelationMode>(relation_mode);
                 auto ranked = s.search(plan, {});
                 return pack_dict(s.pack(ranked, plan));
             },
             nb::arg("text"), nb::arg("at_time") = 0,
             nb::arg("time_mode") = 0, nb::arg("time_start") = 0,
             nb::arg("time_end") = 0, nb::arg("candidate_cap") = 32u,
             nb::arg("expansion_cap") = 1u, nb::arg("token_budget") = size_t{512},
             nb::arg("relation_mode") = 0,
             "Search + minimum-sufficient evidence pack.")
        .def("profile",
             [](const Store& s, int64_t at_time, uint32_t top_k) {
                 auto p = s.profile(at_time, top_k);
                 nb::dict out;
                 out["static_facts"] = results_list(p.static_facts);
                 out["dynamic_facts"] = results_list(p.dynamic_facts);
                 return out;
             },
             nb::arg("at_time") = 0, nb::arg("top_k") = 20)

        // --- persistence ---
        .def("save", &Store::save, nb::arg("path"))
        .def("load", &Store::load, nb::arg("path"))

        // --- introspection ---
        .def_prop_ro("cell_count", &Store::cell_count)
        .def_prop_ro("edge_count", &Store::edge_count)
        .def_prop_ro("entity_count", &Store::entity_count)
        .def_prop_ro("episode_count", &Store::episode_count)
        .def_prop_ro("projection_count", &Store::projection_count);
}