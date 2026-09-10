// 01_prefill.cpp — extract hidden states for one prompt.
//
// The minimal shape of the API: load the model, open a session, arm a capture
// for the prefill, commit one frame, take the result.
//
// Output shape: (1, n_requests, n_embd).
//
// Run:
//   ./build/prefill_example -m model.gguf -p "hello world" -o out.npy
//   ./build/prefill_example -m model.gguf -p "hello world" -o out.npy -l 20,25,29

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/capture.hpp"
#include "ggml_extractor/model.hpp"
#include "ggml_extractor/session.hpp"
#include "ggml_extractor/token_selector.hpp"

using namespace ggml_extractor;

namespace {

std::vector<int> parse_layers(const std::string& csv) {
    std::vector<int> layers;
    std::size_t start = 0;
    while (start <= csv.size()) {
        const std::size_t comma = csv.find(',', start);
        const std::string item = csv.substr(start, comma - start);
        if (!item.empty()) {
            layers.push_back(std::stoi(item));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return layers;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt, output, layer_csv = "20";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* opt) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + opt);
            }
            return argv[i];
        };
        if (arg == "-m") {
            model_path = value("-m");
        } else if (arg == "-p") {
            prompt = value("-p");
        } else if (arg == "-o") {
            output = value("-o");
        } else if (arg == "-l") {
            layer_csv = value("-l");
        } else {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> -p <prompt> -o <out.npy> [-l 20,25,29]\n";
            return arg == "-h" || arg == "--help" ? 0 : 1;
        }
    }
    if (model_path.empty() || prompt.empty() || output.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -m <model.gguf> -p <prompt> -o <out.npy> [-l 20,25,29]\n";
        return 1;
    }

    try {
        auto model = Model::load(model_path);
        Session session(model, SessionOptions{});

        // WHAT to capture — chosen per capture, not per session.
        std::vector<ExtractionRequest> requests{
            {"inp_scaled", TokenSelector::last()},
        };
        for (const int layer : parse_layers(layer_csv)) {
            requests.push_back({"l_out-" + std::to_string(layer), TokenSelector::last()});
        }
        HiddenStateCapture capture(std::move(requests));

        {
            auto armed = session.arm(capture);
            // Logits are requested so the final layer is reachable: llama.cpp
            // gathers l_out-<n_layer-1> down to the output positions.
            session.decode(0, model->tokenize(prompt), /*logits_last=*/true);
            capture.commit_frame();
        }  // disarmed here — the session is back to full-speed inference

        const HiddenStates states = capture.take();
        states.save_npy(output);
        std::cout << "wrote " << output << " shape=(" << states.n_frames << ", "
                  << states.n_requests << ", " << states.n_embd << ")\n";
        for (std::size_t i = 0; i < capture.requests().size(); ++i) {
            std::cout << "  [" << i << "] " << capture.requests()[i].tensor_name << " token="
                      << capture.requests()[i].token.to_string()
                      << " first=" << states.row(0, i)[0] << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
