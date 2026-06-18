#include "llama.h"
#include "src/llama-grammar.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct grammar_case {
    std::string name;
    std::string grammar;
    std::vector<std::vector<llama_token>> fixed_sequences;
    bool full_vocab = false;
    bool expected_build_reject = false;
};

struct divergence {
    std::string grammar_name;
    std::string mode;
    std::string prefix;
    size_t step = 0;
    llama_token token = 0;
    std::string piece;
    bool old_allowed = false;
    bool gss_allowed = false;
    std::string grammar;
};

struct stats {
    size_t grammars_total = 0;
    size_t grammars_built = 0;
    size_t grammars_rejected_by_both = 0;
    size_t sequences_total = 0;
    size_t fixed_sequences = 0;
    size_t random_sequences = 0;
    size_t compared_states = 0;
    size_t token_decisions = 0;
    size_t full_vocab_states = 0;
    size_t sample_states = 0;
    size_t stopped_sequences = 0;
    size_t divergences = 0;
};

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static std::string escape_piece(const std::string & s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (c == '\n') {
            out << "\\n";
        } else if (c == '\r') {
            out << "\\r";
        } else if (c == '\t') {
            out << "\\t";
        } else if (c == '\\') {
            out << "\\\\";
        } else if (c == '"') {
            out << "\\\"";
        } else if (c < 32 || c >= 127) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02x", c);
            out << buf;
        } else {
            out << c;
        }
    }
    return out.str();
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[512];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return {};
    }
    return std::string(buf, buf + n);
}

static std::vector<llama_token> tokenize_or_empty(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, false);
    if (n == 0) {
        return {};
    }
    if (n < 0) {
        n = -n;
    }
    std::vector<llama_token> tokens((size_t) n);
    int got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, false);
    if (got < 0) {
        got = -got;
    }
    tokens.resize((size_t) got);
    return tokens;
}

static std::string prefix_string(const llama_vocab * vocab, const std::vector<llama_token> & prefix) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << prefix[i] << ":'" << escape_piece(token_piece(vocab, prefix[i])) << "'";
    }
    out << "]";
    return out.str();
}

static void add_unique(std::vector<llama_token> & ids, std::set<llama_token> & seen, llama_token id) {
    if (seen.insert(id).second) {
        ids.push_back(id);
    }
}

static std::vector<llama_token> make_full_ids(const llama_vocab * vocab) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> ids;
    ids.reserve((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        ids.push_back((llama_token) i);
    }
    return ids;
}

static std::vector<llama_token> make_sample_ids(
        const llama_vocab * vocab,
        const std::vector<llama_token> & full_ids,
        const std::vector<llama_token> & forced,
        std::mt19937 & rng) {
    std::vector<llama_token> ids;
    std::set<llama_token> seen;
    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (llama_token id = 0; id < std::min(512, n_vocab); ++id) {
        add_unique(ids, seen, id);
    }
    for (llama_token id : forced) {
        if (0 <= id && id < n_vocab) {
            add_unique(ids, seen, id);
        }
    }

    const std::string ascii = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]():,._-+*/= \n\t\"";
    for (unsigned char ch : ascii) {
        auto toks = tokenize_or_empty(vocab, std::string(1, (char) ch));
        for (llama_token id : toks) {
            add_unique(ids, seen, id);
        }
    }

    std::uniform_int_distribution<int> dist(0, n_vocab - 1);
    while (ids.size() < 4096 && ids.size() < full_ids.size()) {
        add_unique(ids, seen, (llama_token) dist(rng));
    }
    return ids;
}

static llama_grammar * init_grammar(const llama_vocab * vocab, const std::string & grammar, bool gss) {
    setenv("LLAMA_GRAMMAR_GSS", gss ? "1" : "0", 1);
    return llama_grammar_init_impl(vocab, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0);
}

static std::vector<uint8_t> accept_vector(const llama_grammar & grammar, const std::vector<llama_token> & ids) {
    std::vector<llama_token_data> data;
    data.reserve(ids.size());
    for (llama_token id : ids) {
        data.push_back({ id, 0.0f, 0.0f });
    }
    llama_token_data_array arr{ data.data(), data.size(), -1, false };
    llama_grammar_apply_impl(grammar, &arr);

    std::vector<uint8_t> allowed;
    allowed.reserve(ids.size());
    for (const auto & tok : data) {
        allowed.push_back(!(std::isinf(tok.logit) && tok.logit < 0.0f));
    }
    return allowed;
}

static bool compare_state(
        const llama_vocab * vocab,
        const grammar_case & gc,
        const std::string & mode,
        const std::vector<llama_token> & ids,
        const std::vector<llama_token> & prefix,
        size_t step,
        llama_grammar & old_grammar,
        llama_grammar & gss_grammar,
        stats & st,
        std::vector<divergence> & divergences,
        std::vector<uint8_t> * accepted_out) {
    const auto old_allowed = accept_vector(old_grammar, ids);
    const auto gss_allowed = accept_vector(gss_grammar, ids);

    st.compared_states++;
    st.token_decisions += ids.size();
    if (gc.full_vocab) {
        st.full_vocab_states++;
    } else {
        st.sample_states++;
    }

    bool ok = true;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (old_allowed[i] != gss_allowed[i]) {
            ok = false;
            st.divergences++;
            if (divergences.size() < 20) {
                divergences.push_back({
                    gc.name,
                    mode,
                    prefix_string(vocab, prefix),
                    step,
                    ids[i],
                    token_piece(vocab, ids[i]),
                    old_allowed[i] != 0,
                    gss_allowed[i] != 0,
                    gc.grammar,
                });
            }
        }
    }

    if (accepted_out != nullptr) {
        *accepted_out = old_allowed;
    }
    return ok;
}

static bool accept_token_both(llama_grammar & old_grammar, llama_grammar & gss_grammar, llama_token token) {
    try {
        llama_grammar_accept_impl(old_grammar, token);
        llama_grammar_accept_impl(gss_grammar, token);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static void run_fixed_sequence(
        const llama_vocab * vocab,
        const grammar_case & gc,
        const std::vector<llama_token> & candidate_ids,
        const std::vector<llama_token> & seq,
        stats & st,
        std::vector<divergence> & divergences) {
    llama_grammar * old_grammar = init_grammar(vocab, gc.grammar, false);
    llama_grammar * gss_grammar = init_grammar(vocab, gc.grammar, true);
    if (old_grammar == nullptr || gss_grammar == nullptr) {
        if (old_grammar != nullptr) llama_grammar_free_impl(old_grammar);
        if (gss_grammar != nullptr) llama_grammar_free_impl(gss_grammar);
        return;
    }

    st.sequences_total++;
    st.fixed_sequences++;
    std::vector<llama_token> prefix;

    for (size_t step = 0; step <= seq.size(); ++step) {
        std::vector<uint8_t> accepted;
        compare_state(vocab, gc, "fixed", candidate_ids, prefix, step, *old_grammar, *gss_grammar, st, divergences, &accepted);
        if (step == seq.size()) {
            break;
        }

        auto it = std::find(candidate_ids.begin(), candidate_ids.end(), seq[step]);
        if (it == candidate_ids.end() || !accepted[(size_t) std::distance(candidate_ids.begin(), it)]) {
            st.stopped_sequences++;
            break;
        }
        if (!accept_token_both(*old_grammar, *gss_grammar, seq[step])) {
            st.stopped_sequences++;
            break;
        }
        prefix.push_back(seq[step]);
    }

    llama_grammar_free_impl(old_grammar);
    llama_grammar_free_impl(gss_grammar);
}

static void run_random_walk(
        const llama_vocab * vocab,
        const grammar_case & gc,
        const std::vector<llama_token> & candidate_ids,
        int max_steps,
        std::mt19937 & rng,
        stats & st,
        std::vector<divergence> & divergences) {
    llama_grammar * old_grammar = init_grammar(vocab, gc.grammar, false);
    llama_grammar * gss_grammar = init_grammar(vocab, gc.grammar, true);
    if (old_grammar == nullptr || gss_grammar == nullptr) {
        if (old_grammar != nullptr) llama_grammar_free_impl(old_grammar);
        if (gss_grammar != nullptr) llama_grammar_free_impl(gss_grammar);
        return;
    }

    st.sequences_total++;
    st.random_sequences++;
    std::vector<llama_token> prefix;

    for (int step = 0; step < max_steps; ++step) {
        std::vector<uint8_t> accepted;
        compare_state(vocab, gc, "random-walk", candidate_ids, prefix, (size_t) step, *old_grammar, *gss_grammar, st, divergences, &accepted);

        std::vector<llama_token> choices;
        for (size_t i = 0; i < candidate_ids.size(); ++i) {
            if (!accepted[i]) {
                continue;
            }
            llama_token id = candidate_ids[i];
            if (llama_vocab_is_eog(vocab, id)) {
                continue;
            }
            const auto piece = token_piece(vocab, id);
            if (piece.empty() || piece[0] == 0) {
                continue;
            }
            choices.push_back(id);
        }
        if (choices.empty()) {
            break;
        }

        std::uniform_int_distribution<size_t> dist(0, choices.size() - 1);
        llama_token chosen = choices[dist(rng)];
        if (!accept_token_both(*old_grammar, *gss_grammar, chosen)) {
            st.stopped_sequences++;
            break;
        }
        prefix.push_back(chosen);
    }

    llama_grammar_free_impl(old_grammar);
    llama_grammar_free_impl(gss_grammar);
}

static std::string random_literal(std::mt19937 & rng) {
    static const char alphabet[] = { 'a', 'b', 'c', 'x', 'y', 'z', '0', '1', '{', '}', '[', ']', ':', ',' };
    std::uniform_int_distribution<size_t> dist(0, sizeof(alphabet) - 1);
    char ch = alphabet[dist(rng)];
    if (ch == '"') {
        return "\"\\\"\"";
    }
    return std::string("\"") + ch + "\"";
}

static std::string random_atom(int rule_index, int n_rules, std::mt19937 & rng) {
    std::uniform_int_distribution<int> kind_dist(0, 8);
    int kind = kind_dist(rng);
    if (kind <= 3 || rule_index + 1 >= n_rules) {
        return random_literal(rng);
    }
    if (kind == 4) {
        return "[abcxyz01]";
    }
    if (kind == 5) {
        return "(" + random_literal(rng) + " | " + random_literal(rng) + ")?";
    }
    if (kind == 6) {
        return random_literal(rng) + "*";
    }
    if (kind == 7) {
        return random_literal(rng) + "+";
    }

    std::uniform_int_distribution<int> ref_dist(rule_index + 1, n_rules - 1);
    std::string ref = "r" + std::to_string(ref_dist(rng));
    std::uniform_int_distribution<int> suffix_dist(0, 4);
    int suffix = suffix_dist(rng);
    if (suffix == 0) return ref;
    if (suffix == 1) return ref + "?";
    if (suffix == 2) return ref + "*";
    return ref;
}

static std::string make_random_grammar(int index, std::mt19937 & rng) {
    std::uniform_int_distribution<int> rules_dist(1, 4);
    const int n_rules = rules_dist(rng);
    std::ostringstream out;
    out << "root ::= r1\n";
    for (int r = 1; r <= n_rules; ++r) {
        std::uniform_int_distribution<int> alts_dist(1, 3);
        int alts = alts_dist(rng);
        out << "r" << r << " ::= ";
        for (int a = 0; a < alts; ++a) {
            if (a != 0) {
                out << " | ";
            }
            std::uniform_int_distribution<int> len_dist(0, 4);
            int len = len_dist(rng);
            if (len == 0) {
                if ((index + r + a) % 5 == 0) {
                    continue;
                }
                len = 1;
            }
            for (int i = 0; i < len; ++i) {
                if (i != 0) {
                    out << " ";
                }
                out << random_atom(r, n_rules + 1, rng);
            }
        }
        out << "\n";
    }
    return out.str();
}

static void write_report(
        const std::string & path,
        const stats & st,
        const std::vector<divergence> & divergences,
        const std::vector<std::string> & corpus_notes,
        size_t full_vocab_size,
        size_t sample_size) {
    std::ofstream out(path);
    out << "# llama.cpp #4218 differential language-equivalence validation\n\n";
    out << "## Result\n\n";
    out << (st.divergences == 0 ? "PASS" : "FAIL") << ": baseline and experimental stack-sharing accept sets matched for every compared token decision.\n\n";
    out << "## Counts\n\n";
    out << "| Metric | Count |\n|---|---:|\n";
    out << "| Grammars total | " << st.grammars_total << " |\n";
    out << "| Grammars built by both engines | " << st.grammars_built << " |\n";
    out << "| Grammars rejected by both engines | " << st.grammars_rejected_by_both << " |\n";
    out << "| Sequence pairs | " << st.sequences_total << " |\n";
    out << "| Fixed sequence pairs | " << st.fixed_sequences << " |\n";
    out << "| Random-walk sequence pairs | " << st.random_sequences << " |\n";
    out << "| Compared generation states | " << st.compared_states << " |\n";
    out << "| Token decisions compared | " << st.token_decisions << " |\n";
    out << "| Full-vocab states | " << st.full_vocab_states << " |\n";
    out << "| Sampled-vocab states | " << st.sample_states << " |\n";
    out << "| Stopped sequences with common rejection/accept exception | " << st.stopped_sequences << " |\n";
    out << "| Divergences | " << st.divergences << " |\n\n";

    out << "Full vocab size: `" << full_vocab_size << "` tokens. Sample pool size: `" << sample_size << "` tokens plus per-sequence forced tokens where needed.\n\n";

    out << "## Corpus\n\n";
    for (const auto & note : corpus_notes) {
        out << "- " << note << "\n";
    }
    out << "\n";

    out << "## Divergences\n\n";
    if (divergences.empty()) {
        out << "None observed.\n\n";
    } else {
        for (const auto & d : divergences) {
            out << "### " << d.grammar_name << " step " << d.step << "\n\n";
            out << "- Mode: `" << d.mode << "`\n";
            out << "- Prefix: `" << d.prefix << "`\n";
            out << "- Token: `" << d.token << "` piece `\"" << escape_piece(d.piece) << "\"`\n";
            out << "- Baseline allowed: `" << (d.old_allowed ? "true" : "false") << "`\n";
            out << "- Experimental allowed: `" << (d.gss_allowed ? "true" : "false") << "`\n\n";
            out << "```gbnf\n" << d.grammar << "\n```\n\n";
        }
    }

    out << "## Reproducibility notes\n\n";
    out << "- Observed: comparison calls the real `llama_grammar_apply_impl()` for both recognizers at every compared state.\n";
    out << "- Observed: baseline and experimental grammars are initialized in-process by setting `LLAMA_GRAMMAR_GSS=0` then `1`; both use the same loaded vocab-only GGUF.\n";
    out << "- Observed: hidden-left-recursion and epsilon-cycle adversarial grammars that the shared parser rejects are counted as common build rejections, not accept-set comparisons.\n";
    out << "- Unknown: sampled-vocab fuzzing is not a formal proof over all possible tokenizations; it is an adversarial differential search over the reported decision count.\n";
    out << "- Unknown: this harness does not by itself prove `CHAR_ALT` multi-range, raw-byte token, clone, partial-UTF-8, or public stack-access compatibility.\n";
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <vocab.gguf> <out.md> <workflow-grammar.gbnf> [workflow-input.txt]\n", argv[0]);
        return 2;
    }

    const std::string vocab_path = argv[1];
    const std::string out_path = argv[2];
    const std::string workflow_grammar = read_file(argv[3]);
    const std::string workflow_input = argc > 4 ? read_file(argv[4]) : "";

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load vocab model: %s\n", vocab_path.c_str());
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::mt19937 rng(0x42184218u);
    const auto full_ids = make_full_ids(vocab);

    std::vector<grammar_case> cases;
    auto add_string_case = [&](const std::string & name, const std::string & grammar, const std::vector<std::string> & strings, bool full) {
        grammar_case gc;
        gc.name = name;
        gc.grammar = grammar;
        gc.full_vocab = full;
        for (const auto & s : strings) {
            gc.fixed_sequences.push_back(tokenize_or_empty(vocab, s));
        }
        cases.push_back(std::move(gc));
    };
    auto add_token_case = [&](const std::string & name, const std::string & grammar, const std::vector<std::vector<llama_token>> & seqs, bool full) {
        grammar_case gc;
        gc.name = name;
        gc.grammar = grammar;
        gc.fixed_sequences = seqs;
        gc.full_vocab = full;
        cases.push_back(std::move(gc));
    };

    add_string_case("test_integer_min0", R"GBNF(
root ::= ([0] | [1-9] [0-9]{0,15}) space
space ::= | " " | "\n"{1,2} [ \t]{0,20}
)GBNF", { "0", "10", "10000", "01" }, true);

    add_string_case("test_expression", R"GBNF(
root ::= expr
expr ::= term ([-+*/] term)*
term ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num ::= [0-9]+ ws
ws ::= [ \t\n]*
)GBNF", { "x+10", "(a+b)*3", "123+456", "x+" }, true);

    add_token_case("test_token_delimiters", R"GBNF(
root ::= <[10]> content <[11]>
content ::= (!<[11]>)*
)GBNF", { {10, 20, 21, 11}, {10, 11}, {20, 11} }, true);

    add_token_case("test_complex_tokens", R"GBNF(
root ::= reasoning+ content tool-call*
reasoning ::= <[10]> (!<[11]>)* <[11]>
content ::= <[20]> (!<[21]>)* <[21]>
tool-call ::= <[12]> name <[13]> args <[14]>
name ::= (!<[13]>)+
args ::= (!<[14]>)*
)GBNF", { {10, 30, 11, 20, 31, 21}, {10, 11, 20, 21, 12, 40, 13, 41, 14} }, true);

    add_string_case("test_special_ellipsis", R"GBNF(
root ::= ... "abc" ...
)GBNF", { "abcabc", "aaaabcccc", "zzzabcqqq" }, true);

    add_string_case("test_quantifier_star", "root ::= \"a\"*\n", { "", "a", "aaaa", "ab" }, true);
    add_string_case("test_exact_repetition", "root ::= [ab]{4}\n", { "aaaa", "abab", "aaa", "aaaaa" }, true);
    add_string_case("test_nullable_repetition_from_tests", "root ::= ( [x]* )*\n", { "", "x", "xx", "y" }, true);

    add_string_case("adversarial_ambiguous_wrappers", R"GBNF(
root ::= nest
nest ::= "a" nest "b" | "a" nest "c" |
)GBNF", { "", "ab", "aabb", "aaabbb", "aaaccc" }, true);

    add_string_case("adversarial_nullable_body_repetition", R"GBNF(
root ::= ("a"?)* "b"
)GBNF", { "b", "ab", "aaab", "a" }, true);

    add_string_case("adversarial_highly_ambiguous", R"GBNF(
root ::= amb "z"
amb ::= "a" amb | "a" amb "b" |
)GBNF", { "z", "az", "aabz", "aaabbz" }, true);

    add_string_case("adversarial_deep_nesting", R"GBNF(
root ::= r1
r1 ::= "a" r2 "z" | "a"
r2 ::= "b" r3 "y" | "b"
r3 ::= "c" r4 "x" | "c"
r4 ::= "d" r5 "w" | "d"
r5 ::= "e" r6 "v" | "e"
r6 ::= "f" r7 "u" | "f"
r7 ::= "g" r8 "t" | "g"
r8 ::= "h"
)GBNF", { "a", "ab", "abcyxz", "abcdefgh", "abcdefghtuvwx yz" }, true);

    add_string_case("adversarial_nullable_helper", R"GBNF(
root ::= maybe "z"
maybe ::= empty | "a" | "a" maybe
empty ::=
)GBNF", { "z", "az", "aaaz", "a" }, true);

    if (!workflow_grammar.empty()) {
        grammar_case gc;
        gc.name = "recursive_pydantic_workflow";
        gc.grammar = workflow_grammar;
        gc.full_vocab = true;
        if (!workflow_input.empty()) {
            gc.fixed_sequences.push_back(tokenize_or_empty(vocab, workflow_input));
        }
        cases.push_back(std::move(gc));
    }

    grammar_case hidden_lr;
    hidden_lr.name = "adversarial_hidden_left_recursion_common_reject";
    hidden_lr.grammar = R"GBNF(
root ::= asdf
asdf ::= "a" | foo "b"
foo ::= "c" | empty asdf "d" | "e"
empty ::= "blah" |
)GBNF";
    hidden_lr.expected_build_reject = true;
    cases.push_back(std::move(hidden_lr));

    grammar_case epsilon_cycle;
    epsilon_cycle.name = "adversarial_epsilon_cycle_common_reject";
    epsilon_cycle.grammar = R"GBNF(
root ::= a "z"
a ::= b |
b ::= a |
)GBNF";
    epsilon_cycle.expected_build_reject = true;
    cases.push_back(std::move(epsilon_cycle));

    for (int i = 0; i < 240; ++i) {
        grammar_case gc;
        gc.name = "fuzz_gbnf_" + std::to_string(i);
        gc.grammar = make_random_grammar(i, rng);
        gc.full_vocab = false;
        cases.push_back(std::move(gc));
    }

    stats st;
    std::vector<divergence> divergences;
    std::vector<std::string> corpus_notes = {
        "Existing-test-style grammars: integer schema, expression grammar, token delimiters, complex token sections, ellipsis, star/exact repetition, nullable repetition from segfault test.",
        "Hand adversarial grammars: ambiguous wrappers, nullable-body repetition, highly ambiguous recursion, deep nesting, nullable helper, hidden left recursion, epsilon cycle.",
        "Recursive-Pydantic workflow grammar loaded from the #4218 validation fixture.",
        "Random fuzz corpus: 240 generated non-left-recursive GBNF grammars, each with 15 random walks.",
    };

    size_t sample_size_observed = 0;
    for (const auto & gc : cases) {
        st.grammars_total++;

        llama_grammar * old_probe = init_grammar(vocab, gc.grammar, false);
        llama_grammar * gss_probe = init_grammar(vocab, gc.grammar, true);
        const bool old_ok = old_probe != nullptr;
        const bool gss_ok = gss_probe != nullptr;
        if (old_probe != nullptr) llama_grammar_free_impl(old_probe);
        if (gss_probe != nullptr) llama_grammar_free_impl(gss_probe);

        if (!old_ok || !gss_ok) {
            if (!old_ok && !gss_ok) {
                st.grammars_rejected_by_both++;
                continue;
            }
            st.divergences++;
            divergences.push_back({ gc.name, "build", "", 0, 0, "", old_ok, gss_ok, gc.grammar });
            continue;
        }
        st.grammars_built++;

        std::vector<llama_token> forced;
        for (const auto & seq : gc.fixed_sequences) {
            forced.insert(forced.end(), seq.begin(), seq.end());
        }

        std::vector<llama_token> candidate_ids = gc.full_vocab ?
            full_ids : make_sample_ids(vocab, full_ids, forced, rng);
        if (!gc.full_vocab) {
            sample_size_observed = std::max(sample_size_observed, candidate_ids.size());
        }

        for (const auto & seq : gc.fixed_sequences) {
            run_fixed_sequence(vocab, gc, candidate_ids, seq, st, divergences);
        }

        const int random_walks = gc.full_vocab ? 3 : 15;
        const int max_steps = gc.full_vocab ? 8 : 10;
        for (int i = 0; i < random_walks; ++i) {
            run_random_walk(vocab, gc, candidate_ids, max_steps, rng, st, divergences);
        }
    }

    write_report(out_path, st, divergences, corpus_notes, full_ids.size(), sample_size_observed);

    std::printf("grammars_total=%zu\n", st.grammars_total);
    std::printf("grammars_built=%zu\n", st.grammars_built);
    std::printf("grammars_rejected_by_both=%zu\n", st.grammars_rejected_by_both);
    std::printf("sequences_total=%zu\n", st.sequences_total);
    std::printf("compared_states=%zu\n", st.compared_states);
    std::printf("token_decisions=%zu\n", st.token_decisions);
    std::printf("divergences=%zu\n", st.divergences);
    std::printf("report=%s\n", out_path.c_str());

    llama_model_free(model);
    llama_backend_free();
    return st.divergences == 0 ? 0 : 1;
}
