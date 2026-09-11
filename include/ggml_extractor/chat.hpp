#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ggml_extractor {

class Model;

/// @brief One conversation turn. Text only — tool calls are not supported.
struct ChatMessage {
    std::string role;  // "system", "user", "assistant", ...
    std::string content;
};

/// @brief Options for ChatFormat::render().
struct ChatOptions {
    /// Append the assistant's opening markers.
    bool add_generation_prompt = true;
    /// Let the model think first. Templates that do not know this flag
    /// ignore it.
    bool enable_thinking = true;
};

/// @brief Renders a conversation with the model's own Jinja template.
///
/// The template comes from the model file's metadata (or an explicit
/// override), is parsed once here, and renders with llama.cpp's own Jinja
/// engine — the same one `--jinja` uses. This replaces hand-written prompt
/// formats, which break on templates with macros (e.g. Gemma 4).
///
/// The result is meant to be tokenized with defaults: a leading BOS the
/// template emits is stripped here because `Model::tokenize` adds it back.
///
/// @note Parsing happens in the constructor and throws on a bad template,
/// so build one ChatFormat and reuse it. Rendering is read-only and may
/// run on any thread.
class ChatFormat {
public:
    /// @param model The model whose metadata template to use.
    /// @param template_override Use this instead of the metadata template.
    /// @throws std::runtime_error when there is no template or it fails
    /// to parse.
    explicit ChatFormat(const Model& model, const std::string& template_override = "");

    ~ChatFormat();
    ChatFormat(const ChatFormat&) = delete;
    ChatFormat& operator=(const ChatFormat&) = delete;
    ChatFormat(ChatFormat&&) = delete;
    ChatFormat& operator=(ChatFormat&&) = delete;

    /// @brief Render `messages` to a prompt string.
    /// @throws std::runtime_error when the template rejects the input
    /// (e.g. it needs typed message content for media).
    std::string render(const std::vector<ChatMessage>& messages,
                       const ChatOptions& options = ChatOptions{}) const;

    /// @brief The template source this renders with.
    const std::string& source() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
