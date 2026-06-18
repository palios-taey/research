#include "llama.h"
#include "src/llama-grammar.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

static std::string read_text(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "failed to open grammar file: %s\n", path);
        std::exit(2);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <vocab.gguf> <grammar.gbnf> [candidate-count] [iters]\n", argv[0]);
        return 2;
    }

    const char * vocab_path = argv[1];
    const std::string grammar_str = read_text(argv[2]);
    const size_t candidate_count = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 128000;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 30;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load vocab model: %s\n", vocab_path);
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const size_t n_vocab = (size_t) llama_vocab_n_tokens(vocab);
    const size_t n_candidates = std::min(candidate_count, n_vocab);
    if (n_candidates < candidate_count) {
        std::fprintf(stderr, "vocab has only %zu tokens; requested %zu\n", n_vocab, candidate_count);
        return 4;
    }

    llama_grammar * grammar = llama_grammar_init_impl(vocab, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (grammar == nullptr) {
        std::fprintf(stderr, "failed to build grammar\n");
        return 5;
    }

    std::vector<llama_token_data> data(n_candidates);
    for (size_t i = 0; i < n_candidates; ++i) {
        data[i] = llama_token_data{ (llama_token) i, 0.0f, 0.0f };
    }
    llama_token_data_array arr{ data.data(), data.size(), -1, false };

    auto reset_logits = [&]() {
        for (auto & tok : data) {
            tok.logit = 0.0f;
        }
    };

    for (int i = 0; i < 5; ++i) {
        reset_logits();
        llama_grammar_apply_impl(*grammar, &arr);
    }

    std::vector<double> us;
    us.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        reset_logits();
        const auto start = std::chrono::steady_clock::now();
        llama_grammar_apply_impl(*grammar, &arr);
        const auto end = std::chrono::steady_clock::now();
        us.push_back((double) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
    }

    size_t rejected = 0;
    for (const auto & tok : data) {
        if (std::isinf(tok.logit) && tok.logit < 0.0f) {
            ++rejected;
        }
    }

    std::sort(us.begin(), us.end());
    const double sum = std::accumulate(us.begin(), us.end(), 0.0);
    const double mean = sum / us.size();
    const double median = us[us.size() / 2];
    const double p95 = us[(size_t) ((us.size() - 1) * 95 / 100)];

    std::printf("vocab_path=%s\n", vocab_path);
    std::printf("n_vocab=%zu\n", n_vocab);
    std::printf("candidate_count=%zu\n", n_candidates);
    std::printf("iters=%d\n", iters);
    std::printf("rejected=%zu\n", rejected);
    std::printf("mean_us=%.3f\n", mean);
    std::printf("median_us=%.3f\n", median);
    std::printf("p95_us=%.3f\n", p95);

    llama_grammar_free_impl(grammar);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
