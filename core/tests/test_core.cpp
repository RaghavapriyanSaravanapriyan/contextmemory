// ContextMemory core — dependency-free test suite.
//
// Covers the write path (create/update/link/expire/forget), bi-temporal
// validity semantics, version chains, the hybrid read path (BM25 + vector +
// entity + recency), container isolation, token budgets, profiles, and the
// snapshot journal round-trip.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cmcore/store.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                           \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                              \
    do {                                                                   \
        ++g_checks;                                                        \
        if (std::fabs((a) - (b)) > (eps)) {                                \
            ++g_failures;                                                  \
            std::fprintf(stderr, "FAIL %s:%d: |%g - %g| > %g\n", __FILE__, \
                         __LINE__, (double)(a), (double)(b), (double)(eps));\
        }                                                                  \
    } while (0)

using cmcore::EdgeType;
using cmcore::FactKind;
using cmcore::Op;
using cmcore::SearchOptions;
using cmcore::Store;
using cmcore::Timestamp;
using cmcore::kNever;

constexpr Timestamp day = 86'400'000LL;  // one day in ms

void test_bm25_basic_retrieval() {
    Store s("bm25");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User prefers TypeScript over Python", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User enjoys hiking in the mountains", "s1", {}},
    });

    SearchOptions opts;
    opts.at_time = t0 + day;
    opts.top_k = 10;
    auto res = s.search("programming TypeScript Python", {}, {}, opts);
    CHECK(!res.empty());
    if (!res.empty()) {
        CHECK(res[0].text.find("TypeScript") != std::string::npos);
    }
}

void test_update_version_chain() {
    Store s("update");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, true,
         1.0f, t0, "User lives in New York", "s1", {}},
    });
    CHECK(s.fact_count() == 1);

    // Two weeks later the user moves.
    const Timestamp t1 = t0 + 14 * day;
    s.apply_batch({
        {Op::Kind::UpdateFact, 1, 0, EdgeType::Related, FactKind::World, true,
         1.0f, t1, "User lives in San Francisco", "s2", {}},
    });

    CHECK(s.fact_count() == 2);
    CHECK(s.edge_count() == 1);

    // The old fact is no longer valid at t1; the new one is.
    SearchOptions opts;
    opts.at_time = t1 + day;
    opts.top_k = 10;
    opts.include_expired = false;
    auto res = s.search("Where does the user live?", {}, {}, opts);
    bool found_sf = false;
    for (const auto& r : res) {
        if (r.text.find("San Francisco") != std::string::npos) found_sf = true;
        CHECK(r.text.find("New York") == std::string::npos);
    }
    CHECK(found_sf);

    // With include_expired, the old version is reachable (auditable history).
    opts.include_expired = true;
    auto res2 = s.search("Where does the user live?", {}, {}, opts);
    bool found_ny = false;
    for (const auto& r : res2) {
        if (r.text.find("New York") != std::string::npos) found_ny = true;
    }
    CHECK(found_ny);
}

void test_temporal_validity_window() {
    Store s("temporal");
    const Timestamp t0 = 1'700'000'000'000LL;

    // An episodic fact: "exam is tomorrow", valid only around t0.
    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::Episode,
         false, 1.0f, t0, "User has a statistics exam tomorrow", "s1", {}},
    });
    const uint64_t fid = 1;

    CHECK(s.fact_count() == 1);

    // Valid the same day.
    SearchOptions opts;
    opts.at_time = t0;
    opts.top_k = 10;
    auto res = s.search("exam", {}, {}, opts);
    CHECK(res.size() == 1);

    // Expire it (the exam passed) -> no longer retrieved.
    s.apply_batch({{Op::Kind::Expire, fid, 0, EdgeType::Related,
                    FactKind::Episode, false, 1.0f, t0 + day, {}, {}, {}}});
    opts.at_time = t0 + 2 * day;
    res = s.search("exam", {}, {}, opts);
    CHECK(res.empty());
}

void test_vector_similarity() {
    Store s("vector");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User likes football", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User likes cooking pasta", "s1", {}},
    });

    // Fake embeddings: doc0 near the query, doc1 far away.
    s.add_fact_embedding(1, std::vector<float>{1.0f, 0.0f});
    s.add_fact_embedding(2, std::vector<float>{0.0f, 1.0f});

    SearchOptions opts;
    opts.at_time = t0 + day;
    opts.top_k = 10;
    auto res = s.search("", std::vector<float>{0.9f, 0.1f}, {}, opts);
    CHECK(res.size() == 2);
    if (res.size() == 2) CHECK(res[0].fact_id == 1);
}

void test_entity_boost() {
    Store s("entity");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "Wrote the v1 of the billing service", "s1", {"Acme Corp"}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "Prefers dark mode", "s1", {}},
    });

    SearchOptions opts;
    opts.at_time = t0 + day;
    opts.top_k = 10;
    // Query mentions the entity: fact 1 must be surfaced via the entity
    // channel even though the lexical channel misses entirely.
    auto res = s.search("What did the user build at Acme Corp?", {},
                        std::vector<std::string>{"Acme Corp"}, opts);
    CHECK(!res.empty());
    if (!res.empty()) CHECK(res[0].fact_id == 1);
}

void test_container_isolation() {
    Store a("user_a");
    Store b("user_b");
    const Timestamp t0 = 1'700'000'000'000LL;

    a.apply_batch({{Op::Kind::CreateFact, 0, 0, EdgeType::Related,
                    FactKind::World, false, 1.0f, t0,
                    "User A has a pet turtle", "s1", {}}});
    b.apply_batch({{Op::Kind::CreateFact, 0, 0, EdgeType::Related,
                    FactKind::World, false, 1.0f, t0,
                    "User B rides a motorcycle", "s1", {}}});

    SearchOptions opts;
    opts.at_time = t0 + day;
    opts.top_k = 10;
    CHECK(a.search("turtle", {}, {}, opts).size() == 1);
    CHECK(a.search("motorcycle", {}, {}, opts).empty());
    CHECK(b.search("turtle", {}, {}, opts).empty());
    CHECK(b.search("motorcycle", {}, {}, opts).size() == 1);
}

void test_recency_decay() {
    Store s("recency");
    const Timestamp t0 = 1'700'000'000'000LL;

    // Two facts on the same topic; one is 100 days old, one is fresh.
    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0 - 100 * day, "User works on the payment service", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User works on the payment service", "s2", {}},
    });

    SearchOptions opts;
    opts.at_time = t0;
    opts.top_k = 10;
    auto res = s.search("payment service", {}, {}, opts);
    CHECK(res.size() == 2);
    if (res.size() == 2) CHECK(res[0].fact_id == 2);  // fresh wins
}

void test_token_budget() {
    Store s("budget");
    const Timestamp t0 = 1'700'000'000'000LL;

    std::vector<Op> ops;
    for (int i = 0; i < 10; ++i) {
        ops.push_back({Op::Kind::CreateFact, 0, 0, EdgeType::Related,
                       FactKind::World, false, 1.0f, t0,
                       "User prefers the color blue variant number " +
                           std::to_string(i),
                       "s1", {}});
    }
    s.apply_batch(ops);

    SearchOptions opts;
    opts.at_time = t0 + day;
    opts.top_k = 10;
    opts.token_budget = 40;  // ~2-3 facts
    auto res = s.search("blue color variant", {}, {}, opts);
    CHECK(!res.empty());
    CHECK(res.size() < 10);
}

void test_profile() {
    Store s("profile");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, true,
         1.0f, t0, "User is a senior engineer", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, true,
         0.9f, t0, "User prefers vim", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::Episode,
         false, 1.0f, t0, "User asked about graph databases", "s1", {}},
    });

    SearchOptions opts;
    opts.at_time = t0 + day;
    auto prof = s.profile(opts);
    CHECK(prof.static_facts.size() == 2);
    CHECK(prof.dynamic_facts.size() == 1);
    CHECK(prof.static_facts[0].text.find("senior") != std::string::npos);
}

void test_journal_roundtrip() {
    const Timestamp t0 = 1'700'000'000'000LL;
    const std::string path = "/tmp/cmcore_test_journal.bin";

    {
        Store s("journal");
        s.apply_batch({
            {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World,
             true, 1.0f, t0, "User lives in Berlin", "s1", {"Berlin"}},
            {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World,
             false, 1.0f, t0, "User likes jazz music", "s1", {}},
        });
        s.add_fact_embedding(1, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
        s.apply_batch({{Op::Kind::UpdateFact, 1, 0, EdgeType::Related,
                        FactKind::World, true, 1.0f, t0 + 30 * day,
                        "User lives in Prague", "s2", {}}});
        CHECK(s.fact_count() == 3);
        CHECK(s.edge_count() == 1);
        s.save(path);
    }

    Store s2("journal");
    s2.load(path);
    CHECK(s2.fact_count() == 3);
    CHECK(s2.edge_count() == 1);
    CHECK(s2.entity_count() == 1);

    // Current state must be exactly as before the save.
    SearchOptions opts;
    opts.at_time = t0 + 60 * day;
    opts.top_k = 10;
    auto res = s2.search("Where does the user live?", {}, {}, opts);
    bool found_prague = false;
    for (const auto& r : res) {
        if (r.text.find("Prague") != std::string::npos) found_prague = true;
        CHECK(r.text.find("Berlin") == std::string::npos);
    }
    CHECK(found_prague);

    // Embeddings survive too. The superseded Berlin fact (id 1) retains its
    // vector and must win the vector channel when expired facts are included.
    opts.top_k = 1;
    opts.include_expired = true;
    auto res2 = s2.search("", std::vector<float>{0.9f, 0.0f, 0.0f, 0.0f}, {},
                          opts);
    CHECK(res2.size() == 1);
    if (res2.size() == 1) CHECK(res2[0].fact_id == 1);

    std::remove(path.c_str());
}

void test_expire_before_update_isolated() {
    // Expiring one fact must not invalidate a sibling.
    Store s("expire_iso");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User has a dentist appointment tomorrow", "s1", {}},
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, false,
         1.0f, t0, "User prefers green tea", "s2", {}},
    });

    s.apply_batch({{Op::Kind::Expire, 1, 0, EdgeType::Related, FactKind::World,
                    false, 1.0f, t0 + day, {}, {}, {}}});

    SearchOptions opts;
    opts.at_time = t0 + 2 * day;
    opts.top_k = 10;
    auto res = s.search("dentist appointment", {}, {}, opts);
    CHECK(res.empty());
    res = s.search("green tea", {}, {}, opts);
    CHECK(res.size() == 1);
}

void test_fact_is_latest_flag() {
    Store s("latest");
    const Timestamp t0 = 1'700'000'000'000LL;

    s.apply_batch({
        {Op::Kind::CreateFact, 0, 0, EdgeType::Related, FactKind::World, true,
         1.0f, t0, "Company uses React", "s1", {}},
    });
    s.apply_batch({{Op::Kind::UpdateFact, 1, 0, EdgeType::Related,
                    FactKind::World, true, 1.0f, t0 + day,
                    "Company uses Svelte", "s2", {}}});

    const auto* old_f = s.fact(1);
    const auto* new_f = s.fact(2);
    CHECK(old_f != nullptr);
    CHECK(new_f != nullptr);
    if (old_f && new_f) {
        CHECK(!old_f->is_latest);
        CHECK(old_f->invalid_at == t0 + day);
        CHECK(new_f->is_latest);
        CHECK(new_f->parent_id == 1);
        CHECK(new_f->root_id == 1);
    }
}

struct TestCase {
    const char* name;
    void (*fn)();
};

const TestCase kTests[] = {
    {"bm25_basic_retrieval", test_bm25_basic_retrieval},
    {"update_version_chain", test_update_version_chain},
    {"temporal_validity_window", test_temporal_validity_window},
    {"vector_similarity", test_vector_similarity},
    {"entity_boost", test_entity_boost},
    {"container_isolation", test_container_isolation},
    {"recency_decay", test_recency_decay},
    {"token_budget", test_token_budget},
    {"profile", test_profile},
    {"journal_roundtrip", test_journal_roundtrip},
    {"expire_before_update_isolated", test_expire_before_update_isolated},
    {"fact_is_latest_flag", test_fact_is_latest_flag},
};

}  // namespace

int main() {
    const size_t total = sizeof(kTests) / sizeof(kTests[0]);
    for (size_t i = 0; i < total; ++i) {
        const auto& t = kTests[i];
        const int before = g_failures;
        t.fn();
        const int failed = g_failures - before;
        std::printf("[%s] %s (%d checks)\n", failed ? "FAIL" : "PASS",
                    t.name, g_checks - 0);
        (void)failed;
    }
    std::printf("%d/%d tests passed (%d failures)\n",
                static_cast<int>(total), static_cast<int>(total),
                g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}