// 06_workers.cpp — many requests at once, one process, no HTTP layer.
//
// A job file lists one request per line, tab-separated:
//
//   <mode>\t<image path or "-">\t<prompt>
//
// mode is one of:
//   ocr         read the prompt (and image) and answer with text
//   ocr-states  same, but also record every generated token's hidden states
//   embed       no text out: record the prompt's hidden states only
//
// A small crew of worker threads (default 2, like -np 2) pulls jobs from a
// shared queue. Each worker owns a Session, a Sampler and (with --draft) a
// Speculative object; the Model, the ChatFormat and the projector are shared.
// Image work is locked per call — the projector is loaded once, not once per
// worker — while text-only jobs never touch the lock.
//
// Outputs, one set per job, numbered by input line (0001, 0002, ...):
//   <out>/NNNN.txt  generated text (ocr modes)
//   <out>/NNNN.npy  hidden states (embed and ocr-states modes)
//   <out>/NNNN.err  error text, when a job fails (the rest still run)
//
// Thinking budget applies to sequential generation (ocr-states, and ocr
// without --draft). Draft generation is greedy and unbounded by design.
//
// Run:
//   ./build/workers_example -m model.gguf --mmproj mmproj.gguf \
//       --draft mtp.gguf --jobs jobs.txt -o out -w 2

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml_extractor/capture.hpp"
#include "ggml_extractor/chat.hpp"
#include "ggml_extractor/model.hpp"
#include "ggml_extractor/multimodal.hpp"
#include "ggml_extractor/sampler.hpp"
#include "ggml_extractor/session.hpp"
#include "ggml_extractor/speculative.hpp"
#include "ggml_extractor/thinking_budget.hpp"
#include "ggml_extractor/token_selector.hpp"

using namespace ggml_extractor;

namespace {

struct Job {
    std::size_t index;  // 1-based input line
    std::string mode;   // ocr | ocr-states | embed
    std::string image;  // empty = text only
    std::string prompt;
};

struct Config {
    std::string model_path;
    std::string mmproj_path;
    std::string draft_path;
    std::string jobs_path;
    std::string out_dir = "out";
    int workers = 2;
    uint32_t n_ctx = 8192;
    uint32_t n_batch = 0;   // 0 = same as n_ctx
    uint32_t n_ubatch = 0;  // 0 = llama.cpp default; raise past --image-max-tokens
    int n_predict = 64;
    std::string layer_csv = "20";
    int thinking_budget = -1;
    SamplerOptions sampling;
    MultimodalOptions multimodal;
};

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

std::vector<Job> read_jobs(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open jobs file: " + path);
    }
    std::vector<Job> jobs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::size_t first = line.find('\t');
        const std::size_t second = line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            throw std::runtime_error("bad job line (want mode<TAB>image<TAB>prompt): " + line);
        }
        Job job;
        job.index = jobs.size() + 1;
        job.mode = line.substr(0, first);
        job.image = line.substr(first + 1, second - first - 1);
        job.prompt = line.substr(second + 1);
        if (job.image == "-") {
            job.image.clear();
        }
        if (job.mode != "ocr" && job.mode != "ocr-states" && job.mode != "embed") {
            throw std::runtime_error("unknown job mode: " + job.mode);
        }
        jobs.push_back(job);
    }
    if (jobs.empty()) {
        throw std::runtime_error("no jobs in " + path);
    }
    return jobs;
}

std::string numbered(const std::string& dir, std::size_t index, const char* ext) {
    char name[16];
    std::snprintf(name, sizeof(name), "%04zu%s", index, ext);
    return (std::filesystem::path(dir) / name).string();
}

// Sequential generation with optional recording and thinking budget.
// The prompt is already in the sequence; logits for it are on.
std::string generate_text(Session& session, int32_t seq, int n_predict, Sampler& sampler,
                          const ThinkingBudgetOptions& budget_options,
                          HiddenStateCapture* capture) {
    Model& model = session.model();
    ThinkingBudget budget(budget_options);
    std::string text;
    std::optional<Session::Armed> armed;
    if (capture != nullptr) {
        capture->begin_frame();  // drop the prefill frame, keep the KV cache
        armed.emplace(session.arm(*capture));
    }
    for (int step = 0; step < n_predict; ++step) {
        const int32_t next = sampler.sample(session);
        if (model.is_eog(next)) {
            break;
        }
        session.decode_one(seq, next);
        const std::string piece = model.token_to_piece(next);
        text += piece;
        if (budget.observe(piece)) {
            session.decode(seq, model.tokenize(budget_options.close_marker,
                                               /*add_special=*/false,
                                               /*parse_special=*/true));
            budget.injected();
        }
        if (capture != nullptr) {
            capture->commit_frame();  // one frame per generated token
        }
    }
    return text;
}

struct Shared {
    std::shared_ptr<Model> model;
    std::unique_ptr<ChatFormat> chat;
    std::unique_ptr<Multimodal> vision;  // null without --mmproj
    std::string marker;
    std::mutex vision_mutex;  // image work, one call at a time
    Config config;
    std::vector<int> layers;
};

void prefill(Session& session, Shared& shared, int32_t seq, const Job& job,
             bool logits_last) {
    std::string text = job.prompt;
    if (!job.image.empty()) {
        text = shared.marker + "\n" + text;
    }
    const std::string formatted = shared.chat->render({{"user", text}});
    if (job.image.empty()) {
        session.decode(seq, shared.model->tokenize(formatted), logits_last);
        return;
    }
    std::lock_guard<std::mutex> lock(shared.vision_mutex);
    shared.vision->eval(session, seq, formatted, {job.image}, logits_last);
}

void run_job(Shared& shared, Session& session, Sampler& sampler, Speculative* draft,
             const Job& job) {
    const int32_t seq = 0;
    session.reset(seq);
    const std::string txt_path = numbered(shared.config.out_dir, job.index, ".txt");
    const std::string npy_path = numbered(shared.config.out_dir, job.index, ".npy");

    auto capture_requests = [&] {
        std::vector<ExtractionRequest> requests;
        for (const int layer : shared.layers) {
            requests.push_back({"l_out-" + std::to_string(layer),
                                job.mode == "embed" ? TokenSelector::last()
                                                    : TokenSelector::generated()});
        }
        return requests;
    };

    if (job.mode == "embed") {
        HiddenStateCapture capture(capture_requests());
        {
            auto armed = session.arm(capture);
            prefill(session, shared, seq, job, /*logits_last=*/true);
            capture.commit_frame();
        }
        capture.take().save_npy(npy_path);
        return;
    }

    if (job.mode == "ocr" && draft != nullptr) {
        prefill(session, shared, seq, job, /*logits_last=*/true);
        std::string text;
        draft->generate(seq, shared.config.n_predict, text);
        std::ofstream(txt_path) << text;
        return;
    }

    ThinkingBudgetOptions budget_options;
    budget_options.max_tokens = shared.config.thinking_budget;
    prefill(session, shared, seq, job, /*logits_last=*/true);
    std::string text;
    if (job.mode == "ocr-states") {
        HiddenStateCapture capture(capture_requests());
        text = generate_text(session, seq, shared.config.n_predict, sampler, budget_options,
                             &capture);
        capture.take().save_npy(npy_path);
    } else {
        text = generate_text(session, seq, shared.config.n_predict, sampler, budget_options,
                             nullptr);
    }
    std::ofstream(txt_path) << text;
}

void worker(Shared& shared, std::deque<Job>& queue, std::mutex& queue_mutex,
            std::condition_variable& queue_cv, bool& done, bool& failed) {
    SessionOptions options;
    options.n_ctx = shared.config.n_ctx;
    options.n_batch = shared.config.n_batch;
    options.n_ubatch = shared.config.n_ubatch;
    options.n_seq_max = 1;
    Session session(shared.model, options);
    Sampler sampler(shared.config.sampling);
    std::unique_ptr<Speculative> draft;
    if (!shared.config.draft_path.empty()) {
        draft.reset(new Speculative(session, shared.config.draft_path));
    }
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&] { return done || !queue.empty(); });
            if (queue.empty()) {
                return;
            }
            job = queue.front();
            queue.pop_front();
        }
        try {
            run_job(shared, session, sampler, draft.get(), job);
        } catch (const std::exception& ex) {
            std::ofstream(numbered(shared.config.out_dir, job.index, ".err"))
                << ex.what() << "\n";
            failed = true;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config config;
    config.sampling.temp = 0.0f;  // greedy unless asked (deterministic)
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* opt) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + opt);
            }
            return argv[i];
        };
        if (arg == "-m") {
            config.model_path = value("-m");
        } else if (arg == "--mmproj") {
            config.mmproj_path = value("--mmproj");
        } else if (arg == "--draft") {
            config.draft_path = value("--draft");
        } else if (arg == "--jobs") {
            config.jobs_path = value("--jobs");
        } else if (arg == "-o") {
            config.out_dir = value("-o");
        } else if (arg == "-w") {
            config.workers = std::stoi(value("-w"));
        } else if (arg == "-c") {
            config.n_ctx = static_cast<uint32_t>(std::stoul(value("-c")));
        } else if (arg == "--batch") {
            config.n_batch = static_cast<uint32_t>(std::stoul(value("--batch")));
        } else if (arg == "--ubatch") {
            config.n_ubatch = static_cast<uint32_t>(std::stoul(value("--ubatch")));
        } else if (arg == "-n") {
            config.n_predict = std::stoi(value("-n"));
        } else if (arg == "-l") {
            config.layer_csv = value("-l");
        } else if (arg == "--thinking-budget") {
            config.thinking_budget = std::stoi(value("--thinking-budget"));
        } else if (arg == "--top-k") {
            config.sampling.top_k = std::stoi(value("--top-k"));
        } else if (arg == "--top-p") {
            config.sampling.top_p = std::stof(value("--top-p"));
        } else if (arg == "--temp") {
            config.sampling.temp = std::stof(value("--temp"));
        } else if (arg == "--seed") {
            config.sampling.seed = static_cast<uint32_t>(std::stoul(value("--seed")));
        } else if (arg == "--image-min-tokens") {
            config.multimodal.image_min_tokens = std::stoi(value("--image-min-tokens"));
        } else if (arg == "--image-max-tokens") {
            config.multimodal.image_max_tokens = std::stoi(value("--image-max-tokens"));
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> --jobs <jobs.txt> [options]\n"
                         "  jobs file: one 'mode<TAB>image-or--<TAB>prompt' per line\n"
                         "  modes: ocr | ocr-states | embed\n";
            return 0;
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return 1;
        }
    }
    if (config.model_path.empty() || config.jobs_path.empty() || config.workers <= 0 ||
        config.n_predict <= 0) {
        std::cerr << "Usage: " << argv[0] << " -m <model.gguf> --jobs <jobs.txt> [options]\n";
        return 1;
    }

    try {
        const std::vector<Job> jobs = read_jobs(config.jobs_path);
        for (const Job& job : jobs) {
            if (!job.image.empty() && config.mmproj_path.empty()) {
                throw std::runtime_error("job " + std::to_string(job.index) +
                                         " needs an image but no --mmproj was given");
            }
        }
        std::filesystem::create_directories(config.out_dir);

        Shared shared;
        shared.config = config;
        shared.layers = parse_layers(config.layer_csv);
        shared.model = Model::load(config.model_path);
        shared.chat.reset(new ChatFormat(*shared.model));
        if (!config.mmproj_path.empty()) {
            shared.vision.reset(new Multimodal(shared.model, config.mmproj_path,
                                               config.multimodal));
            if (!shared.vision->supports_vision()) {
                throw std::runtime_error("this projector does not accept images");
            }
            shared.marker = shared.vision->marker();
        }

        std::deque<Job> queue(jobs.begin(), jobs.end());
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool done = false;
        bool failed = false;
        std::vector<std::thread> threads;
        for (int w = 0; w < config.workers; ++w) {
            threads.emplace_back(worker, std::ref(shared), std::ref(queue),
                                 std::ref(queue_mutex), std::ref(queue_cv), std::ref(done),
                                 std::ref(failed));
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            done = true;
        }
        queue_cv.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
        std::cout << "done: " << jobs.size() << " jobs, " << config.workers << " workers"
                  << (failed ? " (SOME FAILED, see .err files)" : "") << "\n";
        return failed ? 1 : 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
