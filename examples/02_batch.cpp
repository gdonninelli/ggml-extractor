// 02_batch.cpp — many prompts through one session, one output file.
//
// Shows the point of the refactor: the model is loaded once, the context is
// created once, and each prompt reuses both. reset(seq) frees the sequence's
// KV cache between prompts without touching the context.
//
// Output shape: (n_prompts, n_requests, n_embd), input order.
//
// Run:
//   ./build/batch_example -m model.gguf -o batch.npy -p "first" -p "second"

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
    if (layers.empty()) {
        throw std::runtime_error("no layers requested");
    }
    return layers;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, output, layer_csv = "20";
    std::vector<std::string> prompts;
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
        } else if (arg == "-o") {
            output = value("-o");
        } else if (arg == "-p") {
            prompts.push_back(value("-p"));
        } else if (arg == "-l") {
            layer_csv = value("-l");
        } else {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> -o <out.npy> -p <prompt> [-p <prompt> ...]"
                         " [-l 20,25]\n";
            return arg == "-h" || arg == "--help" ? 0 : 1;
        }
    }
    if (model_path.empty() || output.empty() || prompts.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -m <model.gguf> -o <out.npy> -p <prompt> [-p <prompt> ...]"
                     " [-l 20,25]\n";
        return 1;
    }

    try {
        auto model = Model::load(model_path);
        Session session(model, SessionOptions{});

        // One tensor can appear twice with different token rows. second_last()
        // is asked of the first requested layer only: the *final* layer is
        // gathered down to the rows logits were requested for, so it holds no
        // second-to-last token.
        const std::vector<int> layers = parse_layers(layer_csv);
        std::vector<ExtractionRequest> requests;
        for (const int layer : layers) {
            requests.push_back({"l_out-" + std::to_string(layer), TokenSelector::last()});
        }
        requests.push_back({"l_out-" + std::to_string(layers.front()),
                            TokenSelector::second_last()});
        HiddenStateCapture capture(std::move(requests));

        auto armed = session.arm(capture);
        for (const std::string& prompt : prompts) {
            session.reset(0);  // independent prefill, same context
            session.decode(0, model->tokenize(prompt), /*logits_last=*/true);
            capture.commit_frame();  // one frame per prompt
        }

        const HiddenStates states = capture.take();
        states.save_npy(output);
        std::cout << "wrote " << output << " shape=(" << states.n_frames << ", "
                  << states.n_requests << ", " << states.n_embd << ")\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
