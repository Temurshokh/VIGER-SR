#pragma once

#include <cstddef>
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
