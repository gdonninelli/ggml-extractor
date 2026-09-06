// 04_raw_ggml.cpp — plug-and-play on raw ggml graphs (no LlamaSession).
//
// The extractor is model-agnostic: any loop that visits every ggml_tensor*
// of an evaluated graph can drive it with the same two-phase protocol
// llama.cpp uses for `cb_eval`:
//
//   for (ggml_tensor* t : graph_tensors) extractor.filter(t, /*ask=*/true);
//   // ... evaluate the graph ...
//   for (ggml_tensor* t : graph_tensors) extractor.filter(t, /*ask=*/false);
//
// Or equivalently, pass HiddenStateExtractor::callback as your cb_eval and
// set user_data to the extractor (see HiddenStateExtractor::attach).
//
// This example compiles without running inference; it documents the wiring.

#include <iostream>

#include "ggml_extractor/extractor.hpp"
#include "ggml_extractor/token_selector.hpp"

// Forward declaration only — no link dependency in this example.
struct ggml_tensor;

int main() {
    // WHAT to capture: exact tensor names + per-request token rows.
    ggml_extractor::HiddenStateExtractor extractor({
        {"inp_scaled", ggml_extractor::TokenSelector::last()},
        {"l_out-7", ggml_extractor::TokenSelector::last()},
        {"l_out-7", ggml_extractor::TokenSelector::from_end(1)},
    });

    // Schematic wiring for your own ggml loop:
    //
    //   extractor.begin_frame();
    //   for (ggml_tensor* t : tensors) {
    //       if (extractor.filter(t, /*ask=*/true)) { /* keep t in graph */ }
    //   }
    //   evaluate_graph();
    //   for (ggml_tensor* t : tensors) {
    //       extractor.filter(reinterpret_cast<::ggml_tensor*>(t), /*ask=*/false);
    //   }
    //   extractor.require_frame_complete();
    //   extractor.commit_frame();
    //   extractor.save_npy("raw.npy");  // (frames, requests, embd)

    std::cout << "configured " << extractor.request_count() << " request(s):\n";
    for (std::size_t i = 0; i < extractor.request_count(); ++i) {
        const auto& req = extractor.request(i);
        std::cout << "  [" << i << "] tensor='" << req.tensor_name << "' token="
                  << req.token.to_string() << "\n";
    }
    std::cout << "Wire extractor.filter(t, ask) into your ggml graph loop.\n";
    return 0;
}
