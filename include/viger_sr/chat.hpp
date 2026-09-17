#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace viger::chat {

enum class Role { System, User, Assistant };

struct Message {
    Role role;
    std::string content;
};

class IChatModel {
public:
    virtual ~IChatModel() = default;

    [[nodiscard]] virtual std::string generate(
        const std::vector<Message>& history,
        std::string_view user_message) = 0;
};

// Deterministic local fallback used to validate the conversation/session layer.
// It is intentionally not presented as an LLM.
class FallbackChatModel final : public IChatModel {
public:
    [[nodiscard]] std::string generate(
        const std::vector<Message>& history,
        std::string_view user_message) override;
};

// Adapter for a local OpenAI-compatible inference endpoint such as llama.cpp's
// local server. No VIGER backend is involved; the model stays on the same machine.
class LocalHttpChatModel final : public IChatModel {
public:
    struct Config {
        std::string host = "127.0.0.1";
        std::uint16_t port = 8080;
        std::string endpoint = "/v1/chat/completions";
        std::string model = "local-model";
        std::string api_key;
        int max_tokens = 256;
        float temperature = 0.7f;
        std::size_t max_history_chars = 24000;
        int timeout_ms = 120000;
    };

    explicit LocalHttpChatModel(Config config = {});

    [[nodiscard]] std::string generate(
        const std::vector<Message>& history,
        std::string_view user_message) override;

private:
    Config config_;
};

class ChatSession {
public:
    explicit ChatSession(std::unique_ptr<IChatModel> model);

    [[nodiscard]] std::string send(std::string_view user_message);
    void add_system_message(std::string message);
    void clear();
    [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }

private:
    std::unique_ptr<IChatModel> model_;
    std::vector<Message> history_;
};

} // namespace viger::chat
