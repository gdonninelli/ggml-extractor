#include "ggml_extractor/thinking_budget.hpp"

namespace ggml_extractor {

struct ThinkingBudget::Impl {
    ThinkingBudgetOptions options;
    // Tail of recent text, long enough to catch a marker split across
    // pieces. Rebuilt every call, so per-token work stays flat.
    std::string tail;
    int thinking = 0;
    bool forcing = false;
    bool done = false;
    bool counting = false;
};

ThinkingBudget::ThinkingBudget(const ThinkingBudgetOptions& options)
    : impl_(new Impl()) {
    impl_->options = options;
}

ThinkingBudget::~ThinkingBudget() = default;

void ThinkingBudget::reset() {
    impl_->tail.clear();
    impl_->thinking = 0;
    impl_->forcing = false;
    impl_->done = false;
    impl_->counting = false;
}

bool ThinkingBudget::observe(const std::string& piece) {
    Impl& state = *impl_;
    if (state.done || state.forcing || state.options.max_tokens < 0) {
        return false;
    }
    const std::string& open = state.options.open_marker;
    const std::string& close = state.options.close_marker;
    if (open.empty() || close.empty()) {
        return false;
    }

    state.tail += piece;
    if (!state.counting) {
        const std::size_t at = state.tail.find(open);
        if (at == std::string::npos) {
            if (state.tail.size() > open.size()) {
                state.tail.erase(0, state.tail.size() - open.size());
            }
            return false;
        }
        state.counting = true;
        state.tail = state.tail.substr(at + open.size());
        if (state.tail.find(close) != std::string::npos) {
            state.done = true;  // opened and closed within one piece
            return false;
        }
    } else {
        ++state.thinking;
    }

    if (state.tail.find(close) != std::string::npos) {
        state.done = true;  // the model closed the block itself
        return false;
    }
    if (state.thinking > state.options.max_tokens) {
        state.forcing = true;
        return true;
    }
    if (state.tail.size() > close.size()) {
        state.tail.erase(0, state.tail.size() - close.size());
    }
    return false;
}

void ThinkingBudget::injected() {
    impl_->done = true;
    impl_->forcing = false;
}

bool ThinkingBudget::active() const {
    return impl_->counting && !impl_->done;
}

int ThinkingBudget::thinking_tokens() const {
    return impl_->thinking;
}

}  // namespace ggml_extractor
