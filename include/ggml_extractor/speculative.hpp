#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ggml_extractor {

class Session;

/// @brief Options for Speculative.
struct SpeculativeOptions {
    /// Draft tokens proposed per round. Must fit the session batch size
    /// (rounds verify `n_drafts + 1` tokens at once). Upstream default is 3.
    int n_drafts = 3;
    /// GPU layers for the draft weights. `-1` = as many as fit.
    int32_t n_gpu_layers = -1;
};

/// @brief Faster generation using a separate MTP draft head (e.g.
/// `--model-draft mtp-*.gguf --spec-type draft-mtp` in llama-server).
///
/// Each round the draft head proposes a few tokens and the main model
/// checks them all in one go; only matching tokens are kept. Output is
/// identical to greedy sequential generation, reached in fewer rounds.
///
/// Scope, honestly stated:
/// - Only shared-memory MTP heads (Gemma-style: the draft context shares
///   the session's memory) are supported; anything else throws.
/// - Draft rounds pick greedily. For top-k/top-p output, generate
///   sequentially with Sampler instead.
/// - One sequence per call; the session must outlive this object.
///
/// Capture rule (agreed, enforced): generation with the draft runs only
/// while no capture is armed. `generate()` throws when one is — record
/// the prompt with the draft off the table, or record generated tokens
/// with sequential generation.
class Speculative {
public:
    /// @param session Target session. Must outlive this object. Turning
    /// the draft on enables extra model outputs on its context; build
    /// this before the prefill so the first round is primed.
    /// @param draft_path MTP head file for this model.
    /// @throws std::runtime_error when the file is not an MTP head, the
    /// head does not share this session's memory, or the batch is too
    /// small for a round.
    Speculative(Session& session, const std::string& draft_path,
                const SpeculativeOptions& options = SpeculativeOptions{});

    ~Speculative();
    Speculative(const Speculative&) = delete;
    Speculative& operator=(const Speculative&) = delete;
    Speculative(Speculative&&) = delete;
    Speculative& operator=(Speculative&&) = delete;

    /// @brief Append up to `n_predict` generated tokens to `out`.
    ///
    /// Starts with one plain step (which also primes the draft), then
    /// runs draft-and-check rounds. The session's memory holds exactly
    /// the generated tokens afterwards, so plain generation can continue.
    /// @return Tokens appended (stops early on end-of-generation).
    /// @throws std::runtime_error when a capture is armed or a decode fails.
    int generate(int32_t seq, int n_predict, std::string& out);

    /// @brief MTP head count of the draft file.
    int n_mtp_layers() const;
    /// @brief Draft tokens proposed so far / matching ones kept.
    int64_t drafted() const;
    int64_t matched() const;
    /// @brief matched / drafted. -1 when nothing proposed yet.
    double acceptance_rate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
