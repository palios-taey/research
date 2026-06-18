#include "llama.h"
#include "src/llama-grammar.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct case_def {
    std::string cls;
    std::string name;
    std::string grammar;
    std::vector<std::string> seeds;
};

struct class_stats {
    size_t grammars = 0;
    size_t built_both = 0;
    size_t rejected_both = 0;
    size_t build_mismatch = 0;
    size_t sequences = 0;
    size_t states = 0;
    size_t decisions = 0;
    size_t divergences = 0;
};

struct total_stats {
    size_t grammars = 0;
    size_t built_both = 0;
    size_t rejected_both = 0;
    size_t build_mismatch = 0;
    size_t sequences = 0;
    size_t states = 0;
    size_t decisions = 0;
    size_t divergences = 0;
};

struct repro {
    std::string cls;
    std::string name;
    std::string grammar;
    std::string prefix;
    size_t step = 0;
    llama_token token = 0;
    std::string piece;
    bool old_allowed = false;
    bool gss_allowed = false;
};

static std::string escape_text(const std::string & s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c == '\\') out << "\\\\";
        else if (c == '"') out << "\\\"";
        else if (c < 32 || c >= 127) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02x", c);
            out << buf;
        } else {
            out << c;
        }
    }
    return out.str();
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    char buf[512];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n <= 0) return {};
    return std::string(buf, buf + n);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s) {
    int n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), nullptr, 0, false, false);
    if (n < 0) n = -n;
    if (n == 0) return {};
    std::vector<llama_token> out((size_t) n);
    int got = llama_tokenize(vocab, s.data(), (int32_t) s.size(), out.data(), n, false, false);
    if (got < 0) got = -got;
    out.resize((size_t) got);
    return out;
}

static std::string prefix_desc(const llama_vocab * vocab, const std::vector<llama_token> & prefix) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (i) out << ", ";
        out << prefix[i] << ":'" << escape_text(piece(vocab, prefix[i])) << "'";
    }
    out << "]";
    return out.str();
}

static llama_grammar * make_grammar(const llama_vocab * vocab, const std::string & grammar, bool gss) {
    setenv("LLAMA_GRAMMAR_GSS", gss ? "1" : "0", 1);
    return llama_grammar_init_impl(vocab, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0);
}

static std::vector<uint8_t> accepts(const llama_grammar & grammar, const std::vector<llama_token> & vocab_ids) {
    std::vector<llama_token_data> data;
    data.reserve(vocab_ids.size());
    for (llama_token id : vocab_ids) {
        data.push_back({ id, 0.0f, 0.0f });
    }
    llama_token_data_array arr{ data.data(), data.size(), -1, false };
    llama_grammar_apply_impl(grammar, &arr);
    std::vector<uint8_t> ok;
    ok.reserve(data.size());
    for (const auto & t : data) {
        ok.push_back(!(std::isinf(t.logit) && t.logit < 0.0f));
    }
    return ok;
}

static bool accept_one(llama_grammar & old_g, llama_grammar & gss_g, llama_token token) {
    try {
        llama_grammar_accept_impl(old_g, token);
        llama_grammar_accept_impl(gss_g, token);
        return true;
    } catch (...) {
        return false;
    }
}

static std::vector<llama_token> full_vocab_ids(const llama_vocab * vocab) {
    const int n = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> ids;
    ids.reserve((size_t) n);
    for (int i = 0; i < n; ++i) ids.push_back((llama_token) i);
    return ids;
}

static void add_stats(class_stats & cs, total_stats & ts, size_t states, size_t decisions, size_t divs) {
    cs.states += states;
    cs.decisions += decisions;
    cs.divergences += divs;
    ts.states += states;
    ts.decisions += decisions;
    ts.divergences += divs;
}

static size_t compare_state(
        const llama_vocab * vocab,
        const case_def & c,
        const std::vector<llama_token> & ids,
        const std::vector<llama_token> & prefix,
        size_t step,
        llama_grammar & old_g,
        llama_grammar & gss_g,
        std::vector<repro> & repros,
        std::vector<uint8_t> * old_out) {
    const auto old_ok = accepts(old_g, ids);
    const auto gss_ok = accepts(gss_g, ids);
    size_t divs = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (old_ok[i] == gss_ok[i]) continue;
        divs++;
        if (repros.size() < 32) {
            repros.push_back({
                c.cls, c.name, c.grammar, prefix_desc(vocab, prefix), step, ids[i],
                piece(vocab, ids[i]), old_ok[i] != 0, gss_ok[i] != 0,
            });
        }
    }
    if (old_out) *old_out = old_ok;
    return divs;
}

static void run_sequence(
        const llama_vocab * vocab,
        const case_def & c,
        const std::vector<llama_token> & ids,
        const std::vector<llama_token> & seq,
        class_stats & cs,
        total_stats & ts,
        std::vector<repro> & repros) {
    llama_grammar * old_g = make_grammar(vocab, c.grammar, false);
    llama_grammar * gss_g = make_grammar(vocab, c.grammar, true);
    if (!old_g || !gss_g) {
        if (old_g) llama_grammar_free_impl(old_g);
        if (gss_g) llama_grammar_free_impl(gss_g);
        return;
    }
    cs.sequences++;
    ts.sequences++;

    std::vector<llama_token> prefix;
    for (size_t step = 0; step <= seq.size(); ++step) {
        std::vector<uint8_t> old_ok;
        const size_t divs = compare_state(vocab, c, ids, prefix, step, *old_g, *gss_g, repros, &old_ok);
        add_stats(cs, ts, 1, ids.size(), divs);
        if (step == seq.size()) break;
        auto it = std::find(ids.begin(), ids.end(), seq[step]);
        if (it == ids.end()) break;
        const size_t idx = (size_t) std::distance(ids.begin(), it);
        if (!old_ok[idx]) break;
        if (!accept_one(*old_g, *gss_g, seq[step])) break;
        prefix.push_back(seq[step]);
    }
    llama_grammar_free_impl(old_g);
    llama_grammar_free_impl(gss_g);
}

static void run_walk(
        const llama_vocab * vocab,
        const case_def & c,
        const std::vector<llama_token> & ids,
        std::mt19937 & rng,
        class_stats & cs,
        total_stats & ts,
        std::vector<repro> & repros) {
    llama_grammar * old_g = make_grammar(vocab, c.grammar, false);
    llama_grammar * gss_g = make_grammar(vocab, c.grammar, true);
    if (!old_g || !gss_g) {
        if (old_g) llama_grammar_free_impl(old_g);
        if (gss_g) llama_grammar_free_impl(gss_g);
        return;
    }
    cs.sequences++;
    ts.sequences++;

    std::vector<llama_token> prefix;
    for (size_t step = 0; step < 5; ++step) {
        std::vector<uint8_t> old_ok;
        const size_t divs = compare_state(vocab, c, ids, prefix, step, *old_g, *gss_g, repros, &old_ok);
        add_stats(cs, ts, 1, ids.size(), divs);

        std::vector<llama_token> choices;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!old_ok[i]) continue;
            if (llama_vocab_is_eog(vocab, ids[i])) continue;
            const auto p = piece(vocab, ids[i]);
            if (p.empty() || p[0] == 0) continue;
            choices.push_back(ids[i]);
        }
        if (choices.empty()) break;
        std::uniform_int_distribution<size_t> dist(0, choices.size() - 1);
        const llama_token chosen = choices[dist(rng)];
        if (!accept_one(*old_g, *gss_g, chosen)) break;
        prefix.push_back(chosen);
    }

    llama_grammar_free_impl(old_g);
    llama_grammar_free_impl(gss_g);
}

static std::string lit(char ch) {
    if (ch == '\\') return "\"\\\\\"";
    if (ch == '"') return "\"\\\"\"";
    return std::string("\"") + ch + "\"";
}

static std::vector<case_def> generate_cases() {
    std::vector<case_def> out;
    const std::vector<char> letters = { 'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y', 'z', '0', '1' };
    const std::vector<char> suffix = { 'u', 'v', 'w', 'x', 'y', 'z' };

    auto add = [&](std::string cls, std::string name, std::string grammar, std::vector<std::string> seeds) {
        out.push_back({ std::move(cls), std::move(name), std::move(grammar), std::move(seeds) });
    };

    for (int i = 0; i < 420; ++i) {
        char p = letters[(size_t) i % letters.size()];
        char q = letters[(size_t) (i / 3 + 2) % letters.size()];
        char r = suffix[(size_t) i % suffix.size()];
        char s = suffix[(size_t) (i + 1) % suffix.size()];
        int breadth = 2 + (i % 5);
        std::ostringstream g;
        g << "root ::= ";
        for (int b = 0; b < breadth; ++b) {
            if (b) g << " | ";
            g << "entry shared " << lit((b % 2) ? r : s);
        }
        g << "\nentry ::= " << lit(p) << " | " << lit(p) << " optional | optional " << lit(p) << "\n";
        g << "optional ::= | " << lit(q) << "\n";
        g << "shared ::= | " << lit('m') << " | optional " << lit('n') << "\n";
        add("contingent-pop", "contingent_pop_" + std::to_string(i), g.str(), {
            std::string(1, p) + "m" + r,
            std::string(1, p) + q + "n" + s,
            std::string(1, p) + r,
        });
    }

    for (int i = 0; i < 420; ++i) {
        char a = letters[(size_t) i % letters.size()];
        char b = letters[(size_t) (i + 1) % letters.size()];
        char c = letters[(size_t) (i + 2) % letters.size()];
        char x = suffix[(size_t) i % suffix.size()];
        char y = suffix[(size_t) (i + 2) % suffix.size()];
        if (i % 4 == 0) {
            add("deep-mutual-indirect-recursion", "indirect_left_recursion_reject_" + std::to_string(i),
                "root ::= A\nA ::= B " + lit(x) + " | " + lit(a) + "\nB ::= A " + lit(y) + " | " + lit(b) + "\n",
                { std::string(1, a), std::string(1, b) + x });
        } else {
            std::ostringstream g;
            g << "root ::= A\n";
            g << "A ::= " << lit(a) << " B " << lit(x) << " | " << lit(a) << "\n";
            g << "B ::= " << lit(b) << " C " << lit(y) << " | " << lit(b) << "\n";
            g << "C ::= " << lit(c) << " A " << lit('q') << " | " << lit(c) << "\n";
            add("deep-mutual-indirect-recursion", "consuming_cycle_" + std::to_string(i), g.str(), {
                std::string(1, a),
                std::string() + a + b + x,
                std::string() + a + b + c + a + "q" + y + x,
            });
        }
    }

    for (int i = 0; i < 420; ++i) {
        char z = suffix[(size_t) i % suffix.size()];
        if (i % 3 == 0) {
            add("nullable-epsilon-cycles", "epsilon_cycle_reject_" + std::to_string(i),
                "root ::= A " + lit(z) + "\nA ::= | B\nB ::= | A\n",
                { std::string(1, z), std::string("a") + z });
        } else {
            char a = letters[(size_t) i % letters.size()];
            char b = letters[(size_t) (i + 2) % letters.size()];
            std::ostringstream g;
            g << "root ::= left A mid B " << lit(z) << "\n";
            g << "left ::= | " << lit(a) << "\n";
            g << "mid ::= | " << lit('m') << "\n";
            g << "A ::= | " << lit(a) << " | left\n";
            g << "B ::= | " << lit(b) << " | A mid\n";
            add("nullable-epsilon-cycles", "nullable_middle_" + std::to_string(i), g.str(), {
                std::string(1, z),
                std::string() + a + "m" + b + z,
                std::string() + a + a + "m" + z,
            });
        }
    }

    for (int i = 0; i < 420; ++i) {
        char a = letters[(size_t) i % letters.size()];
        char b = letters[(size_t) (i + 1) % letters.size()];
        char end = suffix[(size_t) (i + 3) % suffix.size()];
        std::ostringstream g;
        g << "root ::= (X)* (Y)+ " << lit(end) << "\n";
        g << "X ::= | " << lit(a) << " | " << lit(a) << " X\n";
        g << "Y ::= | " << lit(b) << "\n";
        add("nullable-bodies-in-repetition", "nullable_repetition_" + std::to_string(i), g.str(), {
            std::string(1, end),
            std::string() + a + end,
            std::string() + a + a + b + end,
        });
    }

    for (int i = 0; i < 420; ++i) {
        int breadth = 4 + (i % 8);
        char base = letters[(size_t) i % letters.size()];
        char tail = suffix[(size_t) i % suffix.size()];
        std::ostringstream g;
        g << "root ::= ";
        for (int j = 0; j < breadth; ++j) {
            if (j) g << " | ";
            g << "alt" << j << " common";
        }
        g << "\n";
        for (int j = 0; j < breadth; ++j) {
            g << "alt" << j << " ::= " << lit(base) << " | " << lit(base) << " noise" << j << " | noise" << j << " " << lit(base) << "\n";
            g << "noise" << j << " ::= | " << lit(letters[(size_t) (i + j + 1) % letters.size()]) << "\n";
        }
        g << "common ::= " << lit('r') << " " << lit(tail) << " | " << lit(tail) << "\n";
        add("highly-ambiguous-reconvergence", "reconverge_" + std::to_string(i), g.str(), {
            std::string() + base + tail,
            std::string() + base + "r" + tail,
            std::string() + letters[(size_t) (i + 1) % letters.size()] + base + tail,
        });
    }

    for (int i = 0; i < 420; ++i) {
        int depth = 3 + (i % 7);
        char open = (i % 2) ? '[' : '{';
        char close = (i % 2) ? ']' : '}';
        std::ostringstream g;
        g << "root ::= start payload0 finish\n";
        g << "start ::= " << lit(open) << "\n";
        for (int d = 0; d < depth; ++d) {
            g << "payload" << d << " ::= " << lit(letters[(size_t) (i + d) % letters.size()]) << " payload" << (d + 1) << " " << lit(suffix[(size_t) (i + d) % suffix.size()]) << " | payload" << (d + 1) << "\n";
        }
        g << "payload" << depth << " ::= | " << lit('m') << "\n";
        g << "finish ::= " << lit(close) << "\n";
        add("long-range-dependency", "long_range_" + std::to_string(i), g.str(), {
            std::string() + open + close,
            std::string() + open + "m" + close,
            std::string() + open + letters[(size_t) i % letters.size()] + "m" + suffix[(size_t) i % suffix.size()] + close,
        });
    }

    return out;
}

static void write_report(
        const std::string & path,
        const std::string & vocab_path,
        size_t vocab_size,
        const total_stats & ts,
        const std::map<std::string, class_stats> & by_class,
        const std::vector<repro> & repros) {
    std::ofstream out(path);
    out << "# llama.cpp #4218 independent falsification\n\n";
    out << "## Result\n\n";
    out << (ts.divergences == 0 ? "PASS" : "FAIL") << ": attempted to falsify baseline-vs-experimental language equivalence with a taxonomy-driven adversarial corpus; divergences observed: `" << ts.divergences << "`.\n\n";
    out << "## Run context\n\n";
    out << "- Checkout: `llama.cpp` source tree used for the run\n";
    out << "- Experimental branch: local stack-sharing prototype\n";
    out << "- Commit under test: `26a78fc34afe7e5b93af58ee8b88c268df9569b9`\n";
    out << "- Harness source: `harnesses/independent_falsify.cpp`\n";
    out << "- Harness binary: `<build-dir>/independent_falsify`\n";
    out << "- Vocab oracle universe: `" << vocab_path << "` (`" << vocab_size << "` tokens, full accept-set compared at every state)\n\n";
    out << "## Totals\n\n";
    out << "| Metric | Count |\n|---|---:|\n";
    out << "| Taxonomy grammars generated | " << ts.grammars << " |\n";
    out << "| Built by both engines | " << ts.built_both << " |\n";
    out << "| Rejected upfront by both engines | " << ts.rejected_both << " |\n";
    out << "| Build mismatches | " << ts.build_mismatch << " |\n";
    out << "| Token sequence walks | " << ts.sequences << " |\n";
    out << "| Compared generation states | " << ts.states << " |\n";
    out << "| Full-vocab token decisions compared | " << ts.decisions << " |\n";
    out << "| Divergences | " << ts.divergences << " |\n\n";
    out << "## Per Taxonomy Class\n\n";
    out << "| Class | Grammars | Built both | Rejected both | Build mismatch | Sequences | States | Decisions | Divergences |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto & [name, cs] : by_class) {
        out << "| " << name << " | " << cs.grammars << " | " << cs.built_both << " | " << cs.rejected_both << " | " << cs.build_mismatch
            << " | " << cs.sequences << " | " << cs.states << " | " << cs.decisions << " | " << cs.divergences << " |\n";
    }
    out << "\n";
    out << "## Divergence Repros\n\n";
    if (repros.empty()) {
        out << "None observed.\n\n";
    } else {
        for (const auto & r : repros) {
            out << "### " << r.cls << " / " << r.name << " step " << r.step << "\n\n";
            out << "- Prefix: `" << r.prefix << "`\n";
            out << "- Token: `" << r.token << "` piece `\"" << escape_text(r.piece) << "\"`\n";
            out << "- Baseline allowed: `" << (r.old_allowed ? "true" : "false") << "`\n";
            out << "- Experimental allowed: `" << (r.gss_allowed ? "true" : "false") << "`\n\n";
            out << "```gbnf\n" << r.grammar << "\n```\n\n";
        }
    }
    out << "## Reproducibility notes\n\n";
    out << "- Observed: the harness compares full-vocab `llama_grammar_apply_impl()` accept sets for `LLAMA_GRAMMAR_GSS=0` and `LLAMA_GRAMMAR_GSS=1` at every walked generation state.\n";
    out << "- Observed: the generator in this file is taxonomy-template based and does not reuse the prior `equivalence_harness.cpp` generator.\n";
    out << "- Observed: parser-rejected patterns are reported as upfront common rejects, not treated as equivalence evidence for runtime acceptance.\n";
    out << "- Unknown: this is adversarial falsification, not a formal language-equivalence proof.\n";
    out << "- Unknown: this harness does not by itself prove `CHAR_ALT` multi-range, raw-byte token, clone, partial-UTF-8, or public stack-access compatibility.\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <vocab.gguf> <report.md>\n", argv[0]);
        return 2;
    }

    const std::string vocab_path = argv[1];
    const std::string report_path = argv[2];

    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path.c_str(), params);
    if (!model) {
        std::fprintf(stderr, "failed to load vocab %s\n", vocab_path.c_str());
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto ids = full_vocab_ids(vocab);

    std::mt19937 rng(0x4218f001u);
    const auto cases = generate_cases();
    total_stats ts;
    std::map<std::string, class_stats> by_class;
    std::vector<repro> repros;

    for (const auto & c : cases) {
        auto & cs = by_class[c.cls];
        cs.grammars++;
        ts.grammars++;

        llama_grammar * old_probe = make_grammar(vocab, c.grammar, false);
        llama_grammar * gss_probe = make_grammar(vocab, c.grammar, true);
        const bool old_ok = old_probe != nullptr;
        const bool gss_ok = gss_probe != nullptr;
        if (old_probe) llama_grammar_free_impl(old_probe);
        if (gss_probe) llama_grammar_free_impl(gss_probe);

        if (!old_ok || !gss_ok) {
            if (!old_ok && !gss_ok) {
                cs.rejected_both++;
                ts.rejected_both++;
            } else {
                cs.build_mismatch++;
                ts.build_mismatch++;
                cs.divergences++;
                ts.divergences++;
                if (repros.size() < 32) {
                    repros.push_back({ c.cls, c.name, c.grammar, "build", 0, 0, "", old_ok, gss_ok });
                }
            }
            continue;
        }

        cs.built_both++;
        ts.built_both++;

        for (const auto & seed : c.seeds) {
            run_sequence(vocab, c, ids, tokenize(vocab, seed), cs, ts, repros);
        }
        run_walk(vocab, c, ids, rng, cs, ts, repros);
    }

    write_report(report_path, vocab_path, ids.size(), ts, by_class, repros);
    std::printf("grammars=%zu\n", ts.grammars);
    std::printf("built_both=%zu\n", ts.built_both);
    std::printf("rejected_both=%zu\n", ts.rejected_both);
    std::printf("build_mismatch=%zu\n", ts.build_mismatch);
    std::printf("sequences=%zu\n", ts.sequences);
    std::printf("states=%zu\n", ts.states);
    std::printf("decisions=%zu\n", ts.decisions);
    std::printf("divergences=%zu\n", ts.divergences);
    std::printf("report=%s\n", report_path.c_str());

    llama_model_free(model);
    llama_backend_free();
    return ts.divergences == 0 ? 0 : 1;
}
