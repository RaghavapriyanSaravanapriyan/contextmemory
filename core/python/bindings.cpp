// ContextMemory core — nanobind Python bindings.
//
// The bound surface mirrors the public API of the memory layer: a Store with
// write ops (add/update/link/expire/forget), embeddings, hybrid search,
// profiles, and snapshot persistence. The LLM orchestration never touches the
// core directly; it drives these ops.

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
    d["fact_id"] = r.fact_id;
    d["text"] = r.text;
    d["score"] = r.score;
    d["kind"] = static_cast<int>(r.kind);
    d["is_static"] = r.is_static;
    d["confidence"] = r.confidence;
    d["valid_from"] = r.valid_from;
    d["invalid_at"] = r.invalid_at;
    d["source_ref"] = r.source_ref;
    d["root_id"] = r.root_id;
    return d;
}

nb::list results_list(const std::vector<SearchResult>& results) {
    nb::list out;
    for (const auto& r : results) out.append(result_dict(r));
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "ContextMemory core engine (C++)";

    nb::class_<Store>(m, "Store")
        .def(nb::init<std::string>(), nb::arg("container_tag") = std::string(),
             "Create a store scoped to a container tag.")
        .def("add_fact",
             [](Store& s, const std::string& text, int kind, bool is_static,
                float confidence, int64_t ts, const std::string& ref,
                const std::vector<std::string>& entities) {
                 return s.add_fact(text, static_cast<FactKind>(kind),
                                   is_static, confidence, ts, ref, entities);
             },
             nb::arg("text"), nb::arg("kind") = 0, nb::arg("is_static") = false,
             nb::arg("confidence") = 1.0f, nb::arg("ts") = 0,
             nb::arg("ref") = std::string(),
             nb::arg("entities") = std::vector<std::string>(),
             "Create a fact; returns its id.")
        .def("update_fact", &Store::update_fact, nb::arg("fact_id"),
             nb::arg("text"), nb::arg("ts"),
             nb::arg("entities") = std::vector<std::string>(),
             "Version an existing fact; returns the new fact id.")
        .def("link",
             [](Store& s, int edge_type, uint64_t from, uint64_t to,
                int64_t ts) {
                 s.link(static_cast<EdgeType>(edge_type), from, to, ts);
             },
             nb::arg("edge_type"), nb::arg("from"), nb::arg("to"),
             nb::arg("ts") = 0)
        .def("expire", [](Store& s, uint64_t fact_id, int64_t ts) {
                 Op op;
                 op.kind = Op::Kind::Expire;
                 op.fact_id = fact_id;
                 op.ts = ts;
                 s.apply(op);
             },
             nb::arg("fact_id"), nb::arg("ts"))
        .def("forget", [](Store& s, uint64_t fact_id, int64_t ts) {
                 Op op;
                 op.kind = Op::Kind::Forget;
                 op.fact_id = fact_id;
                 op.ts = ts;
                 s.apply(op);
             },
             nb::arg("fact_id"), nb::arg("ts"))
        .def("set_confidence", [](Store& s, uint64_t fact_id, float value) {
                 Op op;
                 op.kind = Op::Kind::SetConfidence;
                 op.fact_id = fact_id;
                 op.confidence = value;
                 s.apply(op);
             },
             nb::arg("fact_id"), nb::arg("value"))
        .def("add_embedding",
             [](Store& s, uint64_t fact_id, const std::vector<float>& vec) {
                 s.add_fact_embedding(fact_id, vec);
             },
             nb::arg("fact_id"), nb::arg("vector"))
        .def("search",
             [](Store& s, const std::string& text,
                const std::vector<float>& query_vec,
                const std::vector<std::string>& query_entities, int64_t at_time,
                uint32_t top_k, size_t token_budget, bool include_expired,
                uint32_t expand_depth) {
                 SearchOptions opts;
                 opts.at_time = at_time;
                 opts.top_k = top_k;
                 opts.token_budget = token_budget;
                 opts.include_expired = include_expired;
                 opts.expand_depth = expand_depth;
                 return results_list(s.search(text, query_vec, query_entities,
                                              opts));
             },
             nb::arg("text"), nb::arg("query_vec") = std::vector<float>(),
             nb::arg("query_entities") = std::vector<std::string>(),
             nb::arg("at_time") = 0, nb::arg("top_k") = 15,
             nb::arg("token_budget") = size_t{700}, nb::arg("include_expired") = false,
             nb::arg("expand_depth") = 0)
        .def("profile",
             [](Store& s, int64_t at_time, uint32_t top_k) {
                 SearchOptions opts;
                 opts.at_time = at_time;
                 opts.top_k = top_k;
                 auto p = s.profile(opts);
                 nb::dict out;
                 out["static_facts"] = results_list(p.static_facts);
                 out["dynamic_facts"] = results_list(p.dynamic_facts);
                 return out;
             },
             nb::arg("at_time") = 0, nb::arg("top_k") = 20)
        .def("save", &Store::save, nb::arg("path"))
        .def("load", &Store::load, nb::arg("path"))
        .def_prop_ro("fact_count", &Store::fact_count)
        .def_prop_ro("edge_count", &Store::edge_count)
        .def_prop_ro("entity_count", &Store::entity_count);
}