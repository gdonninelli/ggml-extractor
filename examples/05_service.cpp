// 05_service.cpp — one process, one model in memory, several different jobs.
//
// This is the long-running-service shape: the model file is read once, the
// llama_context is created once, and the same context alternates between
//
//   stage 1  plain text generation        (nothing armed — full speed)
//   stage 2  hidden-state extraction      (armed, text prompt)
//   stage 3  image question answering     (nothing armed, via libmtmd)
//   stage 4  hidden-state extraction      (armed, image prompt)
//
// Nothing is reloaded between stages. reset(seq) frees a sequence's KV cache;
// the context and the weights stay put.
//
// Stages 3 and 4 need a vision projector, which llama.cpp always keeps in a
// separate `mmproj-*.gguf` — a text-only model has none, and those stages are
// compiled out entirely when the library is built with
// -DGGML_EXTRACTOR_MULTIMODAL=OFF.
//
// Run (text model):
//   ./build/service_example -m gemma4-12b.gguf -n 48 -l 20,29 -o out
//       -p "The capital of Italy is"
//
// Run (vision model, multimodal build):
//   ./build/service_example -m gemma4.gguf --mmproj mmproj-gemma4.gguf
//       --image photo.jpg --image-prompt "What is in this picture?"
//       -p "The capital of Italy is" -n 48 -l 20,29 -o out

#include "llama.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/capture.hpp"
#include "ggml_extractor/model.hpp"
#include "ggml_extractor/session.hpp"
#include "ggml_extractor/token_selector.hpp"

#ifdef GGML_EXTRACTOR_MULTIMODAL
#include <memory>

#include "ggml_extractor/multimodal.hpp"
#endif

using namespace ggml_extractor;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string text_prompt = "The capital of Italy is";
    std::string image_prompt = "Describe this image in one sentence.";
    std::string output_prefix;
    std::string layer_csv = "20";
    int n_predict = 32;
    uint32_t n_ctx = 4096;
    bool help = false;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " -m <model.gguf> [options]\n\n"
        << "  -m <path>            GGUF model (required)\n"
        << "  -p <text>            text prompt for stages 1 and 2\n"
        << "  -l <a,b,c>           layers to capture, e.g. 20,29 (default 20)\n"
        << "  -n <count>           tokens to generate (default 32)\n"
        << "  -c <count>           context size (default 4096)\n"
        << "  -o <prefix>          write <prefix>-text.npy / <prefix>-image.npy\n"
#ifdef GGML_EXTRACTOR_MULTIMODAL
        << "  --mmproj <path>      vision projector, enables stages 3 and 4\n"
        << "  --image <path>       image file (required with --mmproj)\n"
        << "  --image-prompt <t>   question to ask about the image\n"
#else
        << "\nThis is a text-only build: the image stages are compiled out.\n"
        << "Rebuild with -DGGML_EXTRACTOR_MULTIMODAL=ON to enable them.\n"
#endif
        << "\nLayer indices must be < the model's n_layer, which is printed at\n"
           "startup. inp_scaled and l_out-<n> are Gemma-family tensor names.\n";
}

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

std::vector<ExtractionRequest> build_requests(const std::vector<int>& layers,
                                              TokenSelector token) {
    std::vector<ExtractionRequest> requests;
    requests.reserve(layers.size() + 1);
    requests.push_back({"inp_scaled", token});
    for (const int layer : layers) {
        requests.push_back({"l_out-" + std::to_string(layer), token});
    }
    return requests;
}

/// Wrap a user turn in the model's chat format, so the demo produces sensible
/// completions instead of raw text continuation.
///
/// llama_chat_apply_template() is not a Jinja engine — it pattern-matches a
/// fixed list of known templates and returns -1 for anything else. Gemma 4
/// ships an 18 KB Jinja template with macros, so it lands in that -1 case and
/// we emit Gemma's turn markers ourselves. Production code that must handle
/// arbitrary models wants a real Jinja renderer (llama.cpp's
/// common_chat_templates, which this project does not build).
std::string as_chat_turn(const Model& model, const std::string& content) {
    const char* tmpl = llama_model_chat_template(model.handle(), nullptr);
    if (tmpl != nullptr) {
        llama_chat_message message{"user", content.c_str()};
        std::vector<char> buffer(content.size() * 2 + 1024);
        int32_t written = llama_chat_apply_template(tmpl, &message, 1, /*add_ass=*/true,
                                                    buffer.data(),
                                                    static_cast<int32_t>(buffer.size()));
        if (written > static_cast<int32_t>(buffer.size())) {
            buffer.resize(static_cast<std::size_t>(written));
            written = llama_chat_apply_template(tmpl, &message, 1, /*add_ass=*/true,
                                                buffer.data(),
                                                static_cast<int32_t>(buffer.size()));
        }
        if (written > 0) {
            return std::string(buffer.data(), static_cast<std::size_t>(written));
        }
    }
    // Hand-rolled fallbacks. The markers are control tokens, so the prompt
    // must be tokenized with parse_special = true; BOS comes from the
    // tokenizer's add_special.
    const std::string architecture = model.architecture();
    if (architecture.rfind("gemma4", 0) == 0) {
        // Gemma 4 replaced Gemma 2/3's <start_of_turn>/<end_of_turn> with
        // <|turn>...<turn|> (<turn|> is EOS), and added reasoning channels.
        // Closing an empty thought channel is how the model's own template
        // suppresses reasoning, so the answer arrives directly.
        return "<|turn>user\n" + content +
               "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    }
    if (architecture.rfind("gemma", 0) == 0) {
        return "<start_of_turn>user\n" + content +
               "<end_of_turn>\n<start_of_turn>model\n";
    }
    return content;
}

/// Greedy continuation from whatever is already in `seq`.
std::string generate(Session& session, int32_t seq, int n_predict) {
    const Model& model = session.model();
    std::string text;
    for (int step = 0; step < n_predict; ++step) {
        const int32_t next = session.sample_greedy();
        if (model.is_eog(next)) {
            break;
        }
        session.decode_one(seq, next);
        text += model.token_to_piece(next);
    }
    return text;
}

void report(const HiddenStateCapture& capture, const HiddenStates& states,
            const std::string& output_path) {
    std::cout << "  shape=(" << states.n_frames << ", " << states.n_requests << ", "
              << states.n_embd << ")\n";
    for (std::size_t i = 0; i < capture.requests().size(); ++i) {
        std::cout << "    " << capture.requests()[i].tensor_name
                  << " token=" << capture.requests()[i].token.to_string()
                  << " first3=[" << states.row(0, i)[0] << ", " << states.row(0, i)[1]
                  << ", " << states.row(0, i)[2] << "]\n";
    }
    if (!output_path.empty()) {
        states.save_npy(output_path);
        std::cout << "  wrote " << output_path << "\n";
    }
}

int64_t elapsed_ms(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start)
        .count();
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* opt) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + opt);
            }
            return argv[i];
        };
        if (arg == "-m") {
            args.model_path = value("-m");
        } else if (arg == "--mmproj") {
            args.mmproj_path = value("--mmproj");
        } else if (arg == "--image") {
            args.image_path = value("--image");
        } else if (arg == "-p") {
            args.text_prompt = value("-p");
        } else if (arg == "--image-prompt") {
            args.image_prompt = value("--image-prompt");
        } else if (arg == "-o") {
            args.output_prefix = value("-o");
        } else if (arg == "-l") {
            args.layer_csv = value("-l");
        } else if (arg == "-n") {
            args.n_predict = std::stoi(value("-n"));
        } else if (arg == "-c") {
            args.n_ctx = static_cast<uint32_t>(std::stoul(value("-c")));
        } else if (arg == "-h" || arg == "--help") {
            args.help = true;
            return args;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (args.model_path.empty()) {
        throw std::runtime_error("-m <model.gguf> is required");
    }
    if (args.n_predict <= 0) {
        throw std::runtime_error("-n must be positive");
    }
    if (args.mmproj_path.empty() != args.image_path.empty()) {
        throw std::runtime_error("--mmproj and --image must be given together");
    }
#ifndef GGML_EXTRACTOR_MULTIMODAL
    if (!args.mmproj_path.empty()) {
        throw std::runtime_error("this build has no multimodal support; rebuild with "
                                 "-DGGML_EXTRACTOR_MULTIMODAL=ON");
    }
#endif
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.help) {
            print_usage(argv[0]);
            return 0;
        }
        const std::vector<int> layers = parse_layers(args.layer_csv);
        const int32_t seq = 0;

        // ---- loaded once, for the whole process lifetime ----------------
        const Clock::time_point load_start = Clock::now();
        auto model = Model::load(args.model_path);
        SessionOptions options;
        options.n_ctx = args.n_ctx;
        Session session(model, options);
        std::cout << "model: " << args.model_path << "\n"
                  << "  arch=" << model->architecture() << " n_layer=" << model->n_layer()
                  << " n_embd=" << model->n_embd() << " n_ctx=" << session.n_ctx()
                  << " (loaded in " << elapsed_ms(load_start) << " ms)\n";

#ifdef GGML_EXTRACTOR_MULTIMODAL
        std::unique_ptr<Multimodal> vision;
        if (!args.mmproj_path.empty()) {
            const Clock::time_point mm_start = Clock::now();
            vision.reset(new Multimodal(model, args.mmproj_path));
            std::cout << "  projector=" << args.mmproj_path
                      << " vision=" << (vision->supports_vision() ? "yes" : "no")
                      << " audio=" << (vision->supports_audio() ? "yes" : "no")
                      << " (loaded in " << elapsed_ms(mm_start) << " ms)\n";
            if (!vision->supports_vision()) {
                throw std::runtime_error("this projector does not accept images");
            }
        }
#endif

        // ---- stage 1: plain text generation, nothing armed --------------
        std::cout << "\n[1] text generation (extraction off)\n";
        {
            const Clock::time_point start = Clock::now();
            session.reset(seq);
            session.decode(seq, model->tokenize(as_chat_turn(*model, args.text_prompt)));
            const std::string answer = generate(session, seq, args.n_predict);
            std::cout << "  armed=" << (session.armed() ? "yes" : "no") << " tokens="
                      << session.n_past(seq) << " in " << elapsed_ms(start) << " ms\n"
                      << "  answer: " << answer << "\n";
        }

        // ---- stage 2: hidden states for a text prompt -------------------
        std::cout << "\n[2] hidden-state extraction, text prompt\n";
        {
            const Clock::time_point start = Clock::now();
            HiddenStateCapture capture(build_requests(layers, TokenSelector::last()));
            session.reset(seq);
            {
                auto armed = session.arm(capture);
                // logits_last stays true so the final layer is reachable:
                // llama.cpp gathers l_out-<n_layer-1> down to the output rows.
                session.decode(seq, model->tokenize(args.text_prompt),
                               /*logits_last=*/true);
                capture.commit_frame();
            }  // disarmed — the context is back to full-speed inference
            const HiddenStates states = capture.take();
            std::cout << "  captured in " << elapsed_ms(start) << " ms\n";
            report(capture, states,
                   args.output_prefix.empty() ? "" : args.output_prefix + "-text.npy");
        }

#ifdef GGML_EXTRACTOR_MULTIMODAL
        if (!vision) {
            std::cout << "\npass --mmproj and --image to run the image stages.\n";
            return 0;
        }

        // ---- stage 3: image question answering, nothing armed ----------
        std::cout << "\n[3] image generation (extraction off)\n";
        const std::string media_prompt =
            as_chat_turn(*model, vision->marker() + "\n" + args.image_prompt);
        {
            const Clock::time_point start = Clock::now();
            session.reset(seq);
            vision->eval(session, seq, media_prompt, {args.image_path});
            const std::string answer = generate(session, seq, args.n_predict);
            std::cout << "  image=" << args.image_path << " tokens="
                      << session.n_past(seq) << " in " << elapsed_ms(start) << " ms\n"
                      << "  answer: " << answer << "\n";
        }

        // ---- stage 4: hidden states for the same image prompt ----------
        std::cout << "\n[4] hidden-state extraction, image prompt\n";
        {
            const Clock::time_point start = Clock::now();
            HiddenStateCapture capture(build_requests(layers, TokenSelector::last()));
            session.reset(seq);
            {
                auto armed = session.arm(capture);
                // eval() decodes the image and text chunks on this context, so
                // the armed capture sees them; the frame ends up holding the
                // last prompt token, after the image has been attended to.
                vision->eval(session, seq, media_prompt, {args.image_path},
                             /*logits_last=*/true);
                capture.commit_frame();
            }
            const HiddenStates states = capture.take();
            std::cout << "  captured in " << elapsed_ms(start) << " ms\n";
            report(capture, states,
                   args.output_prefix.empty() ? "" : args.output_prefix + "-image.npy");
        }

        std::cout << "\nall four stages ran against one model and one context.\n";
#else
        std::cout << "\ntext-only build: the image stages are compiled out "
                     "(-DGGML_EXTRACTOR_MULTIMODAL=ON to include them).\n"
                     "both stages above ran against one model and one context.\n";
#endif
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
