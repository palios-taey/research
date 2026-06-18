#include "src/llama-grammar.h"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_file(const char * path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("failed to open ") + path);
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static std::string printable(unsigned char ch) {
    if (ch == '\n') {
        return "\\n";
    }
    if (ch == '\t') {
        return "\\t";
    }
    if (ch >= 32 && ch < 127) {
        return std::string(1, static_cast<char>(ch));
    }
    char buf[8];
    std::snprintf(buf, sizeof(buf), "\\x%02x", ch);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <grammar.gbnf> <input-string>\n", argv[0]);
        return 2;
    }

    const std::string grammar_src = read_file(argv[1]);
    const std::string input = argv[2];

    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_src.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (grammar == nullptr) {
        std::fprintf(stderr, "failed to initialize grammar\n");
        return 1;
    }

    auto & stacks = llama_grammar_get_stacks(grammar);
    const auto & rules = llama_grammar_get_rules(grammar);
    size_t max_stacks = stacks.size();
    long long total_us = 0;
    long long max_us = 0;
    long long total_reject_us = 0;
    long long max_reject_us = 0;

    const std::array<std::array<uint32_t, 2>, 4> candidate_codepoints = {{
        {{static_cast<uint32_t>('a'), 0}},
        {{static_cast<uint32_t>('b'), 0}},
        {{static_cast<uint32_t>('c'), 0}},
        {{static_cast<uint32_t>('d'), 0}},
    }};
    std::vector<llama_grammar_candidate> candidates;
    for (size_t i = 0; i < candidate_codepoints.size(); ++i) {
        candidates.push_back({i, candidate_codepoints[i].data(), llama_partial_utf8{0, 0}, static_cast<llama_token>(1000 + i)});
    }

    std::printf("idx,char,before_stacks,after_stacks,accept_us,reject4_us,reject4_count\n");
    std::fflush(stdout);
    for (size_t i = 0; i < input.size(); ++i) {
        const std::string piece(1, input[i]);
        const size_t before = stacks.size();
        const auto start = std::chrono::steady_clock::now();
        try {
            llama_grammar_accept_token(*grammar, static_cast<llama_token>(i + 1), piece);
        } catch (const std::exception & e) {
            const auto end = std::chrono::steady_clock::now();
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::printf("%zu,%s,%zu,0,%lld\n", i, printable(static_cast<unsigned char>(input[i])).c_str(), before, static_cast<long long>(us));
            std::fflush(stdout);
            std::fprintf(stderr, "accept failed at idx=%zu char=%s: %s\n", i, printable(static_cast<unsigned char>(input[i])).c_str(), e.what());
            llama_grammar_free_impl(grammar);
            return 1;
        }
        const auto end = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_us += us;
        max_us = std::max(max_us, static_cast<long long>(us));
        max_stacks = std::max(max_stacks, stacks.size());

        long long reject_us = 0;
        size_t reject_count = 0;
        if (!stacks.empty()) {
            const auto reject_start = std::chrono::steady_clock::now();
            auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);
            for (size_t is = 1; is < stacks.size() && !rejects.empty(); ++is) {
                rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[is], rejects);
            }
            const auto reject_end = std::chrono::steady_clock::now();
            reject_us = std::chrono::duration_cast<std::chrono::microseconds>(reject_end - reject_start).count();
            reject_count = rejects.size();
        }
        total_reject_us += reject_us;
        max_reject_us = std::max(max_reject_us, reject_us);

        std::printf("%zu,%s,%zu,%zu,%lld,%lld,%zu\n",
                i,
                printable(static_cast<unsigned char>(input[i])).c_str(),
                before,
                stacks.size(),
                static_cast<long long>(us),
                reject_us,
                reject_count);
        std::fflush(stdout);
    }

    const bool accepts_eof = std::any_of(stacks.begin(), stacks.end(), [](const llama_grammar_stack & stack) {
        return stack.empty();
    });

    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);

    std::fprintf(stderr, "summary input_len=%zu final_stacks=%zu max_stacks=%zu total_accept_us=%lld avg_accept_us=%.3f max_accept_us=%lld total_reject4_us=%lld avg_reject4_us=%.3f max_reject4_us=%lld accepts_eof=%s maxrss_kb=%ld\n",
            input.size(),
            stacks.size(),
            max_stacks,
            total_us,
            input.empty() ? 0.0 : static_cast<double>(total_us) / static_cast<double>(input.size()),
            max_us,
            total_reject_us,
            input.empty() ? 0.0 : static_cast<double>(total_reject_us) / static_cast<double>(input.size()),
            max_reject_us,
            accepts_eof ? "true" : "false",
            usage.ru_maxrss);

    llama_grammar_free_impl(grammar);
    return accepts_eof ? 0 : 3;
}
