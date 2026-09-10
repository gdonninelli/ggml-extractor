// 04_raw_ggml.cpp — drive a capture from a raw ggml graph, no model involved.
//
// HiddenStateCapture only needs the two-phase protocol llama.cpp uses for
// cb_eval: ask ("do you want this tensor?") then copy. Any loop that visits
// the nodes of an evaluated graph can drive it, which makes the capture
// testable without loading a model.
//
// This builds a two-node graph on the CPU backend, names the nodes the way
// llama.cpp names its own, computes it, and captures two rows.
//
// Run:
//   ./build/raw_ggml_example [-o raw.npy]

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/capture.hpp"
#include "ggml_extractor/token_selector.hpp"

using namespace ggml_extractor;

int main(int argc, char** argv) {
    std::string output;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output = argv[++i];
        } else {
            std::cout << "Usage: " << argv[0] << " [-o <out.npy>]\n";
            return arg == "-h" || arg == "--help" ? 0 : 1;
        }
    }

    const int64_t n_embd = 8;
    const int64_t n_tokens = 4;

    ggml_backend_t backend = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    try {
        HiddenStateCapture capture({
            {"inp_scaled", TokenSelector::last()},
            {"l_out-0", TokenSelector::last()},
            {"l_out-0", TokenSelector::second_last()},
        });

        backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("failed to initialise the CPU backend");
        }

        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;  // tensors are allocated by the backend below
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create the ggml context");
        }

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(input, "inp");
        ggml_set_input(input);

        ggml_tensor* scaled = ggml_scale(ctx, input, 2.0f);
        ggml_set_name(scaled, "inp_scaled");

        ggml_tensor* layer_out = ggml_add(ctx, scaled, scaled);
        ggml_set_name(layer_out, "l_out-0");
        ggml_set_output(layer_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, layer_out);

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate graph tensors");
        }

        // Row t of the input is filled with the value t, so the captured rows
        // are easy to eyeball: inp_scaled row t == 2t, l_out-0 row t == 4t.
        std::vector<float> host(static_cast<std::size_t>(n_embd * n_tokens));
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t e = 0; e < n_embd; ++e) {
                host[static_cast<std::size_t>(t * n_embd + e)] = static_cast<float>(t);
            }
        }
        ggml_backend_tensor_set(input, host.data(), 0, host.size() * sizeof(float));

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph computation failed");
        }

        // The two-phase protocol, exactly as llama.cpp's scheduler drives it.
        capture.begin_frame();
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (capture.filter(node, /*ask=*/true)) {
                capture.filter(node, /*ask=*/false);
            }
        }
        capture.commit_frame();

        const HiddenStates states = capture.take();
        std::cout << "captured shape=(" << states.n_frames << ", " << states.n_requests
                  << ", " << states.n_embd << ")\n";
        for (std::size_t i = 0; i < capture.requests().size(); ++i) {
            std::cout << "  [" << i << "] " << capture.requests()[i].tensor_name << " token="
                      << capture.requests()[i].token.to_string()
                      << " value=" << states.row(0, i)[0] << "\n";
        }
        if (!output.empty()) {
            states.save_npy(output);
            std::cout << "wrote " << output << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
    }

    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
    if (backend != nullptr) {
        ggml_backend_free(backend);
    }
    return 0;
}
