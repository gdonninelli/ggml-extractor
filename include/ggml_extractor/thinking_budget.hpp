#pragma once

#include <memory>
#include <string>

namespace ggml_extractor {

/// @brief Options for ThinkingBudget. Defaults fit Gemma 4's thought
/// channel (`<|channel>thought` … `<channel|>`); other models pass their
/// own markers.
struct ThinkingBudgetOptions {
    /// Maximum thinking tokens per answer. `< 0` = unlimited (helper inert).
    /// `0` closes the channel as soon as it opens.
    int max_tokens = -1;
    /// Text that opens a thinking block in the generated output.
    std::string open_marker = "<|channel>thought";
    /// Text that closes one.
    std::string close_marker = "<channel|>";
};

/// @brief Forces a thinking block closed after a token budget.
///
/// Sits in the generation loop, watches each generated piece of text, and
/// reports once when the budget runs out. The caller then decodes the close
/// marker and carries on sampling:
///
/// @code
/// ThinkingBudget budget(options);
/// for (...) {
///     int32_t next = sampler.sample(session);
///     if (model.is_eog(next)) break;
///     session.decode_one(seq, next);
///     if (budget.observe(model.token_to_piece(next))) {
///         session.decode(seq, model.tokenize(budget_close, false, true));
///         budget.injected();
///     }
/// }
/// @endcode
///
/// Matching works on text, so markers split across tokens are found. Only
/// the last few characters are kept, so cost per token is flat.
///
/// An injection is an ordinary decode: under an armed capture it lands in
/// the in-progress frame like any other decode and is overwritten by the
/// next one — committed frames only ever hold sampled tokens.
class ThinkingBudget {
public:
    explicit ThinkingBudget(const ThinkingBudgetOptions& options = ThinkingBudgetOptions{});

    ~ThinkingBudget();
    ThinkingBudget(const ThinkingBudget&) = delete;
    ThinkingBudget& operator=(const ThinkingBudget&) = delete;
    ThinkingBudget(ThinkingBudget&&) = delete;
    ThinkingBudget& operator=(ThinkingBudget&&) = delete;

    /// @brief Start over for a new answer.
    void reset();

    /// @brief Feed one generated token's text.
    /// @return True exactly once per answer: decode the close marker now
    /// and call injected(). False otherwise. Never true when unlimited,
    /// when no thinking block opened, or when the model closed it itself.
    bool observe(const std::string& piece);

    /// @brief Tell the helper the close marker was decoded.
    void injected();

    /// @brief True while inside a thinking block (budget not yet enforced).
    bool active() const;
    /// @brief Thinking tokens counted in the current block.
    int thinking_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
