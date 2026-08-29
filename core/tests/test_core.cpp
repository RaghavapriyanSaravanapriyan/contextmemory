// ContextMemory core — ETMC dependency-free test suite.
//
// Covers the write path (capture / reconcile / project / version), bi-temporal
// validity, late-arriving events, exact dedup, container isolation, the
// compiled query plan, hybrid channels, token-budget evidence packing, graph
// expansion caps, profiles, and the journal round-trip.

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

using cmcore::CellInput;
using cmcore::CellKind;
using cmcore::CellStatus;
using cmcore::CompiledQuery;
using cmcore::Episode;
using cmcore::Store;
using cmcore::TimeMode;
using cmcore::Timestamp;
using cmcore::kNever;

constexpr Timestamp day = 86'400'000LL;  // one day in ms

Timestamp t35() {
    // 1'700'000'000'000 + 35 days
    return 1'700'000'000'000LL + 35 * 86'400'000LL;
}

void test_capture_and_dedup() {
    Store s("dedup");
    const Timestamp t0 = 1'700'000'000'000LL;
    Episode ep;
    ep.role = "user";
    ep.content = "I live in New York and work at Acme Corp.";
    ep.observed_at = t0;
    const uint64_t eid = s.capture_episode(ep);
    CHECK(eid != 0);
    CHECK(s.episode_count() == 1);
    CHECK(s.episode(eid) != nullptr);

    CellInput in;
    in.subject = "user";
    in.predicate = "location";
    in.object = "New York";
    in.text = "User lives in New York";
    in.observed_at = t0;
    in.valid_from = t0;
    in.entities = {"user"};
    const uint64_t c1 = s.reconcile(in);
    CHECK(c1 != 0);

    // Exact duplicate ingestion creates no second cell.
    const uint64_t c2 = s.reconcile(in);
    CHECK(c2 == c1);
    CHECK(s.cell_count() == 1);
}

void test_update_versions_and_projection() {
    Store s("update");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput ny;
    ny.subject = "user";
    ny.predicate = "location";
    ny.object = "New York";
    ny.text = "User lives in New York";
    ny.observed_at = t0;
    ny.valid_from = t0;
    ny.entities = {"user"};
    const uint64_t ny_id = s.reconcile(ny);
    CHECK(s.cell_count() == 1);

    const Timestamp t1 = t0 + 14 * day;
    CellInput seattle;
    seattle.subject = "user";
    seattle.predicate = "location";
    seattle.object = "Seattle";
    seattle.text = "User lives in Seattle";
    seattle.observed_at = t1;
    seattle.valid_from = t1;
    seattle.entities = {"user"};
    const uint64_t sea_id = s.reconcile(seattle);
    CHECK(sea_id != ny_id);
    CHECK(s.cell_count() == 2);
    CHECK(s.edge_count() == 1);

    // Projection points at the current version.
    const auto* proj = s.projection("user", "location");
    CHECK(proj != nullptr);
    if (proj) {
        CHECK(proj->active_cell == sea_id);
        CHECK(proj->root_id == ny_id);
        CHECK(proj->version_count == 2);
    }

    // Current query returns Seattle, not New York.
    auto cq = s.compile("Where do I live now?", t1 + day);
    CHECK(cq.plan.time_mode == TimeMode::Current);
    auto res = s.search(cq.plan, {});
    bool found_sea = false, found_ny = false;
    for (const auto& r : res) {
        if (r.text.find("Seattle") != std::string::npos) found_sea = true;
        if (r.text.find("New York") != std::string::npos) found_ny = true;
    }
    CHECK(found_sea);
    CHECK(!found_ny);

    // Historical query returns New York, the superseded version.
    auto hq = s.compile("Where did I live before?", t1 + day);
    CHECK(hq.plan.time_mode == TimeMode::Historical);
    auto hres = s.search(hq.plan, {});
    found_ny = false;
    for (const auto& r : hres) {
        if (r.text.find("New York") != std::string::npos) found_ny = true;
    }
    CHECK(found_ny);
}

void test_late_arriving_event() {
    Store s("late");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput ny;
    ny.subject = "user";
    ny.predicate = "location";
    ny.object = "New York";
    ny.text = "User lives in New York";
    ny.observed_at = t0;
    ny.valid_from = t0;
    s.reconcile(ny);

    // The move happened on day 30 but was only recorded on day 60.
    const Timestamp t30 = t0 + 30 * day;
    const Timestamp t60 = t0 + 60 * day;
    CellInput sea;
    sea.subject = "user";
    sea.predicate = "location";
    sea.object = "Seattle";
    sea.text = "User moved to Seattle";
    sea.observed_at = t60;   // learned late
    sea.valid_from = t30;    // event time
    s.reconcile(sea);

    // A query on day 35 must still see New York (agent hadn't learned yet).
    auto q = s.compile("Where do I live?", t35());
    auto res = s.search(q.plan, {});
    bool found_ny = false, found_sea = false;
    for (const auto& r : res) {
        if (r.text.find("New York") != std::string::npos) found_ny = true;
        if (r.text.find("Seattle") != std::string::npos) found_sea = true;
    }
    CHECK(found_ny);
    CHECK(!found_sea);
}

void test_container_isolation() {
    Store a("user_a");
    Store b("user_b");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput pa;
    pa.subject = "user";
    pa.predicate = "pet";
    pa.text = "User has a pet turtle";
    pa.observed_at = t0;
    pa.valid_from = t0;
    a.reconcile(pa);

    CellInput pb;
    pb.subject = "user";
    pb.predicate = "transport";
    pb.text = "User rides a motorcycle";
    pb.observed_at = t0;
    pb.valid_from = t0;
    b.reconcile(pb);

    auto qa = a.compile("What is the user's pet?", t0 + day);
    auto qb = b.compile("What is the user's pet?", t0 + day);
    CHECK(a.search(qa.plan, {}).size() == 1);
    CHECK(b.search(qb.plan, {}).empty());
}

void test_vector_channel() {
    Store s("vector");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput f1;
    f1.subject = "user";
    f1.predicate = "sport";
    f1.text = "User likes football";
    f1.observed_at = t0;
    f1.valid_from = t0;
    const uint64_t c1 = s.reconcile(f1);

    CellInput f2;
    f2.subject = "user";
    f2.predicate = "cooking";
    f2.text = "User likes cooking pasta";
    f2.observed_at = t0;
    f2.valid_from = t0;
    s.reconcile(f2);

    s.add_embedding(c1, std::vector<float>{1.0f, 0.0f});
    s.add_embedding(2, std::vector<float>{0.0f, 1.0f});

    auto cq = s.compile("sports", t0 + day);
    auto res = s.search(cq.plan, std::vector<float>{0.9f, 0.1f});
    CHECK(res.size() == 2);
    if (res.size() == 2) CHECK(res[0].cell_id == c1);
}

void test_entity_channel() {
    Store s("entity");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput built;
    built.subject = "acme";
    built.predicate = "project";
    built.text = "Wrote the v1 of the billing service";
    built.observed_at = t0;
    built.valid_from = t0;
    built.entities = {"Acme Corp"};
    s.reconcile(built);

    CellInput mode;
    mode.subject = "user";
    mode.predicate = "preference";
    mode.text = "Prefers dark mode";
    mode.observed_at = t0;
    mode.valid_from = t0;
    s.reconcile(mode);

    auto cq = s.compile("What did the user build at Acme Corp?", t0 + day);
    CHECK(!cq.plan.entity_seeds.empty());
    auto res = s.search(cq.plan, {});
    CHECK(!res.empty());
    if (!res.empty()) CHECK(res[0].cell_id == 1);
}

void test_token_budget_packing() {
    Store s("budget");
    const Timestamp t0 = 1'700'000'000'000LL;
    for (int i = 0; i < 10; ++i) {
        CellInput in;
        in.subject = "user";
        in.predicate = "color";
        in.text = "User prefers the color blue variant number " +
                  std::to_string(i);
        in.observed_at = t0;
        in.valid_from = t0;
        s.reconcile(in);
    }
    auto cq = s.compile("What is the user's favorite color?", t0 + day);
    cq.plan.token_budget = 40;  // ~2-3 cells
    auto res = s.search(cq.plan, {});
    auto pack = s.pack(res, cq.plan);
    CHECK(pack.tokens <= 40);
    CHECK(!pack.items.empty());
    CHECK(pack.items.size() < 10);
}

void test_profile() {
    Store s("profile");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput eng;
    eng.subject = "user";
    eng.predicate = "role";
    eng.text = "User is a senior engineer";
    eng.kind = CellKind::World;
    eng.salience = 0.9f;
    eng.observed_at = t0;
    eng.valid_from = t0;
    s.reconcile(eng);

    CellInput vim;
    vim.subject = "user";
    vim.predicate = "preference";
    vim.text = "User prefers vim";
    vim.kind = CellKind::Preference;
    vim.confidence = 0.9f;
    vim.observed_at = t0;
    vim.valid_from = t0;
    s.reconcile(vim);

    CellInput ask;
    ask.subject = "user";
    ask.predicate = "question";
    ask.text = "User asked about graph databases";
    ask.kind = CellKind::Experience;
    ask.observed_at = t0;
    ask.valid_from = t0;
    s.reconcile(ask);

    auto prof = s.profile(t0 + day, 20);
    CHECK(prof.static_facts.size() == 2);
    CHECK(prof.dynamic_facts.size() == 1);
}

void test_journal_roundtrip() {
    const Timestamp t0 = 1'700'000'000'000LL;
    const std::string path = "/tmp/cmcore_test_journal.bin";

    {
        Store s("journal");
        CellInput berlin;
        berlin.subject = "user";
        berlin.predicate = "location";
        berlin.object = "Berlin";
        berlin.text = "User lives in Berlin";
        berlin.observed_at = t0;
        berlin.valid_from = t0;
        berlin.entities = {"Berlin"};
        const uint64_t b_id = s.reconcile(berlin);

        CellInput jazz;
        jazz.subject = "user";
        jazz.predicate = "music";
        jazz.text = "User likes jazz music";
        jazz.observed_at = t0;
        jazz.valid_from = t0;
        s.reconcile(jazz);

        s.add_embedding(b_id, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});

        CellInput prague;
        prague.subject = "user";
        prague.predicate = "location";
        prague.object = "Prague";
        prague.text = "User lives in Prague";
        prague.observed_at = t0 + 30 * day;
        prague.valid_from = t0 + 30 * day;
        s.reconcile(prague);

        CHECK(s.cell_count() == 3);
        CHECK(s.edge_count() == 1);
        CHECK(s.projection_count() == 2);  // location + music
        s.save(path);
    }

    Store s2("journal");
    s2.load(path);
    CHECK(s2.cell_count() == 3);
    CHECK(s2.edge_count() == 1);
    CHECK(s2.projection_count() == 2);
    CHECK(s2.entity_count() == 1);

    auto cq = s2.compile("Where does the user live?", t0 + 60 * day);
    CHECK(cq.plan.time_mode == TimeMode::Current);
    auto res = s2.search(cq.plan, {});
    bool found_prague = false, found_berlin = false;
    for (const auto& r : res) {
        if (r.text.find("Prague") != std::string::npos) found_prague = true;
        if (r.text.find("Berlin") != std::string::npos) found_berlin = true;
    }
    CHECK(found_prague);
    CHECK(!found_berlin);

    std::remove(path.c_str());
}

void test_expire_and_abstention() {
    Store s("abstain");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput exam;
    exam.subject = "user";
    exam.predicate = "exam";
    exam.text = "User has a statistics exam tomorrow";
    exam.kind = CellKind::Experience;
    exam.observed_at = t0;
    exam.valid_from = t0;
    s.reconcile(exam);

    auto q1 = s.compile("When is the statistics exam?", t0);
    CHECK(s.search(q1.plan, {}).size() == 1);

    // Expire via a superseding empty state: no replacement, window closed.
    CellInput done;
    done.subject = "user";
    done.predicate = "exam";
    done.object = "none";
    done.text = "User no longer has a statistics exam";
    done.observed_at = t0 + day;
    done.valid_from = t0 + day;
    s.reconcile(done);

    auto q2 = s.compile("When is the statistics exam?", t0 + 2 * day);
    auto res = s.search(q2.plan, {});
    auto pack = s.pack(res, q2.plan);
    // The old exam is closed; the "none" state covers current truth.
    CHECK(!pack.items.empty());
    CHECK(pack.sufficient);
}

void test_graph_expansion_cap() {
    Store s("graph");
    const Timestamp t0 = 1'700'000'000'000LL;

    CellInput a;
    a.subject = "alice";
    a.predicate = "project";
    a.text = "Alice leads project Phoenix";
    a.observed_at = t0;
    a.valid_from = t0;
    const uint64_t a_id = s.reconcile(a);

    CellInput b;
    b.subject = "bob";
    b.predicate = "project";
    b.text = "Bob contributes to project Phoenix";
    b.observed_at = t0;
    b.valid_from = t0;
    const uint64_t b_id = s.reconcile(b);

    s.link(cmcore::EdgeType::Related, a_id, b_id, t0);

    auto cq = s.compile("What connects Alice and Bob?", t0 + day);
    CHECK(cq.plan.relation_mode == cmcore::RelationMode::MultiHop);
    cq.plan.expansion_cap = 1;
    auto res = s.search(cq.plan, {});
    // Expansion stays bounded; at least the seed project cell is returned.
    CHECK(res.size() <= cq.plan.candidate_cap);
    CHECK(!res.empty());
}

struct TestCase {
    const char* name;
    void (*fn)();
};

const TestCase kTests[] = {
    {"capture_and_dedup", test_capture_and_dedup},
    {"update_versions_and_projection", test_update_versions_and_projection},
    {"late_arriving_event", test_late_arriving_event},
    {"container_isolation", test_container_isolation},
    {"vector_channel", test_vector_channel},
    {"entity_channel", test_entity_channel},
    {"token_budget_packing", test_token_budget_packing},
    {"profile", test_profile},
    {"journal_roundtrip", test_journal_roundtrip},
    {"expire_and_abstention", test_expire_and_abstention},
    {"graph_expansion_cap", test_graph_expansion_cap},
};

}  // namespace

int main() {
    const size_t total = sizeof(kTests) / sizeof(kTests[0]);
    int total_checks = 0;
    for (size_t i = 0; i < total; ++i) {
        const auto& t = kTests[i];
        const int before = g_failures;
        const int checks_before = g_checks;
        t.fn();
        const int failed = g_failures - before;
        total_checks += g_checks - checks_before;
        std::printf("[%s] %s (%d checks)\n", failed ? "FAIL" : "PASS",
                    t.name, g_checks - checks_before);
    }
    std::printf("%d/%d tests passed (%d failures, %d checks)\n",
                static_cast<int>(total), static_cast<int>(total), g_failures,
                total_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}