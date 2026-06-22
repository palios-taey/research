#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-grammar.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

bool llama_grammar_accepts_token_id_for_test(const llama_grammar * grammar, llama_token token);

struct chart_stats {
    size_t origins = 0;
    size_t sealed = 0;
    size_t current_items = 0;
    size_t stored_items = 0;
    size_t resume_entries = 0;
};

static bool is_end_of_sequence(const llama_grammar_element * pos) {
    return pos->type == LLAMA_GRETYPE_END || pos->type == LLAMA_GRETYPE_ALT;
}

static llama_grammar * build_grammar(const std::string & grammar_str) {
    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
    assert(grammar != nullptr);
    return grammar;
}

static bool grammar_is_complete(const llama_grammar * grammar) {
    for (const llama_grammar_item & item : grammar->chart.back().items) {
        if (item.rule == grammar->start_rule_index && item.origin == 0 &&
                is_end_of_sequence(&grammar->rules[item.rule][item.dot])) {
            return true;
        }
    }

    return false;
}

static bool grammar_allows_eog(const llama_grammar * grammar) {
    return grammar->partial_utf8.n_remain == 0 && grammar_is_complete(grammar);
}

static bool accept_token_piece(llama_grammar * grammar, llama_token token, const std::string & piece) {
    try {
        llama_grammar_accept_token(*grammar, token, piece);
    } catch (const std::runtime_error &) {
        return false;
    }

    return !grammar->chart.back().items.empty();
}

static bool match_string(const std::string & grammar_str, const std::string & input) {
    llama_grammar * grammar = build_grammar(grammar_str);
    bool matched = true;

    for (unsigned char c : input) {
        const std::string piece(1, static_cast<char>(c));
        if (!accept_token_piece(grammar, 0, piece)) {
            matched = false;
            break;
        }
    }

    matched = matched && grammar_is_complete(grammar);
    llama_grammar_free_impl(grammar);
    return matched;
}

static chart_stats get_chart_stats(const llama_grammar * grammar) {
    chart_stats stats;
    stats.origins = grammar->chart.size();
    stats.current_items = grammar->chart.back().items.size();

    for (const llama_grammar_chart_column & column : grammar->chart) {
        if (column.sealed) {
            ++stats.sealed;
        }
        stats.stored_items += column.items.size();
        for (const auto & resume : column.resume) {
            ++stats.resume_entries;
            stats.stored_items += resume.second.items.size();
        }
    }

    return stats;
}

static void print_stats(const char * label, size_t n, const chart_stats & stats) {
    std::fprintf(
            stdout,
            "%s n=%zu origins=%zu sealed=%zu current_items=%zu stored_items=%zu resume_entries=%zu\n",
            label,
            n,
            stats.origins,
            stats.sealed,
            stats.current_items,
            stats.stored_items,
            stats.resume_entries);
    std::fflush(stdout);
}

static chart_stats run_a_star(size_t n) {
    llama_grammar * grammar = build_grammar(R"""(root ::= "a"*)""");

    for (size_t i = 0; i < n; ++i) {
        assert(accept_token_piece(grammar, 0, "a"));
    }

    assert(grammar_is_complete(grammar));
    const chart_stats stats = get_chart_stats(grammar);
    print_stats("astar", n, stats);
    llama_grammar_free_impl(grammar);
    return stats;
}

static void test_a_star_plateau(size_t n) {
    const chart_stats stats = run_a_star(n);

    assert(stats.origins <= 8);
    assert(stats.current_items <= 8);
    assert(stats.stored_items <= 64);
}

static void test_balanced_exact(size_t max_n) {
    const std::string grammar = R"""(root ::= "a" root "b" | "")""";

    assert(match_string(grammar, ""));
    for (size_t n = 1; n <= max_n; ++n) {
        const std::string a(n, 'a');
        const std::string b(n, 'b');

        assert(match_string(grammar, a + b));
        assert(!match_string(grammar, a + std::string(n - 1, 'b')));
        assert(!match_string(grammar, a + std::string(n + 1, 'b')));
    }

    llama_grammar * grammar_state = build_grammar(grammar);
    for (size_t i = 0; i < max_n; ++i) {
        assert(accept_token_piece(grammar_state, 0, "a"));
    }

    const chart_stats stats = get_chart_stats(grammar_state);
    print_stats("balanced-prefix", max_n, stats);
    assert(stats.origins >= max_n / 2);

    llama_grammar_free_impl(grammar_state);
}

static void test_token_not_empty_candidate() {
    llama_grammar * grammar = build_grammar(R"""(root ::= !<[42]>)""");

    assert(!llama_grammar_accepts_token_id_for_test(grammar, 42));
    assert(llama_grammar_accepts_token_id_for_test(grammar, 43));

    std::fprintf(stdout, "token-not-empty-candidate excluded=0 allowed=1\n");
    llama_grammar_free_impl(grammar);
}

static bool rule_is_done(const llama_grammar_rule & rule) {
    return rule.size() == 2 &&
        rule[0].type == LLAMA_GRETYPE_CHAR && rule[0].value == static_cast<uint32_t>('x') &&
        is_end_of_sequence(&rule[1]);
}

static bool rule_is_caller(const llama_grammar_rule & rule, uint32_t done_rule) {
    return rule.size() == 4 &&
        rule[0].type == LLAMA_GRETYPE_CHAR && rule[0].value == static_cast<uint32_t>('x') &&
        rule[1].type == LLAMA_GRETYPE_RULE_REF && rule[1].value == done_rule &&
        rule[2].type == LLAMA_GRETYPE_CHAR && rule[2].value == static_cast<uint32_t>('y') &&
        is_end_of_sequence(&rule[3]);
}

static void test_nullable_utf8_eog_after_partial() {
    llama_grammar * grammar = build_grammar(R"""(root ::= "\u00E9" | "")""");
    assert(accept_token_piece(grammar, 100, std::string("\xC3", 1)));
    assert(grammar->partial_utf8.n_remain > 0);
    assert(!grammar_allows_eog(grammar));

    llama_grammar * completes_e9 = llama_grammar_clone_impl(*grammar);
    assert(accept_token_piece(completes_e9, 101, std::string("\xA9", 1)));
    assert(grammar_allows_eog(completes_e9));

    llama_grammar * completes_a0 = llama_grammar_clone_impl(*grammar);
    assert(!accept_token_piece(completes_a0, 102, std::string("\xA0", 1)));
    assert(!grammar_allows_eog(completes_a0));

    std::fprintf(stdout, "nullable-utf8-eog partial=0 completed=1 wrong=0\n");
    llama_grammar_free_impl(completes_a0);
    llama_grammar_free_impl(completes_e9);
    llama_grammar_free_impl(grammar);
}

static void test_completion_reallocation_stress() {
    constexpr size_t n_callers = 8191;
    std::string grammar_str = R"""(root ::= "q"
done ::= "x"
)""";

    for (size_t i = 0; i < n_callers; ++i) {
        grammar_str += "caller" + std::to_string(i) + R"""( ::= "x" done "y"
)""";
    }

    llama_grammar * grammar = build_grammar(grammar_str);

    uint32_t done_rule = std::numeric_limits<uint32_t>::max();
    for (uint32_t rule_id = 0; rule_id < grammar->rules.size(); ++rule_id) {
        if (rule_is_done(grammar->rules[rule_id])) {
            done_rule = rule_id;
            break;
        }
    }
    assert(done_rule != std::numeric_limits<uint32_t>::max());

    std::vector<uint32_t> caller_rules;
    std::vector<bool> is_caller(grammar->rules.size(), false);
    for (uint32_t rule_id = 0; rule_id < grammar->rules.size(); ++rule_id) {
        if (rule_is_caller(grammar->rules[rule_id], done_rule)) {
            caller_rules.push_back(rule_id);
            is_caller[rule_id] = true;
        }
    }
    assert(caller_rules.size() == n_callers);

    grammar->chart.clear();
    grammar->chart.emplace_back();
    grammar->chart.back().items.reserve(n_callers + 1);
    grammar->chart.back().items.push_back({ done_rule, 0, 1 });
    for (uint32_t caller_rule : caller_rules) {
        grammar->chart.back().items.push_back({ caller_rule, 0, 1 });
    }

    llama_grammar_accept(grammar, static_cast<uint32_t>('x'));

    size_t advanced_callers = 0;
    for (const llama_grammar_item & item : grammar->chart.back().items) {
        if (item.rule < is_caller.size() && is_caller[item.rule] && item.dot == 2 && item.origin == 1) {
            ++advanced_callers;
        }
    }

    assert(advanced_callers == n_callers);
    std::fprintf(stdout, "completion-reallocation-stress callers=%zu advanced=%zu final_items=%zu\n",
            n_callers,
            advanced_callers,
            grammar->chart.back().items.size());

    llama_grammar_free_impl(grammar);
}

int main(int argc, char ** argv) {
    const bool long_run = argc > 1 && std::string(argv[1]) == "--long";

    test_a_star_plateau(long_run ? 100000 : 1000);
    test_balanced_exact(long_run ? 512 : 64);
    test_token_not_empty_candidate();
    test_nullable_utf8_eog_after_partial();
    test_completion_reallocation_stress();

    return 0;
}
