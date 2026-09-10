// 03_generate.cpp — capture the hidden state of every generated token.
//
// The prefill frame is dropped with begin_frame(); after that each
// decode_one + commit_frame pair contributes one frame, so the output rows
// line up with the generated tokens.
//
// Output shape: (n_generated, n_requests, n_embd).
//
// Run:
//   ./build/generate_example -m model.gguf -p "Rome is" -o gen.npy -n 32

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
    std::string model_path, prompt, output, layer_csv = "20";
    int n_predict = 32;
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
        } else if (arg == "-n") {
            n_predict = std::stoi(value("-n"));
        } else if (arg == "-l") {
            layer_csv = value("-l");
        } else {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> -p <prompt> -o <out.npy> [-n <tokens>]"
                         " [-l 20,25]\n";
            return arg == "-h" || arg == "--help" ? 0 : 1;
        }
    }
    if (model_path.empty() || prompt.empty() || output.empty() || n_predict <= 0) {
        std::cerr << "Usage: " << argv[0]
                  << " -m <model.gguf> -p <prompt> -o <out.npy> [-n <tokens>]"
                     " [-l 20,25]\n";
        return 1;
    }

    try {
        auto model = Model::load(model_path);
        SessionOptions options;
        options.n_ctx = 4096;
        Session session(model, options);

        std::vector<ExtractionRequest> requests;
        for (const int layer : parse_layers(layer_csv)) {
            requests.push_back(
                {"l_out-" + std::to_string(layer), TokenSelector::generated()});
        }
        HiddenStateCapture capture(std::move(requests));

        auto armed = session.arm(capture);

        session.decode(0, model->tokenize(prompt));
        capture.begin_frame();  // drop the prefill frame, keep the KV cache

        std::string text;
        for (int step = 0; step < n_predict; ++step) {
            const int32_t next = session.sample_greedy();
            if (model->is_eog(next)) {
                break;
            }
            session.decode_one(0, next);
            capture.commit_frame();  // one frame per generated token
            text += model->token_to_piece(next);
        }

        const HiddenStates states = capture.take();
        states.save_npy(output);
        std::cout << "generated: " << text << "\n";
        std::cout << "wrote " << output << " shape=(" << states.n_frames << ", "
                  << states.n_requests << ", " << states.n_embd << ")\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
