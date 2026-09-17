#include "viger_sr/chat.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace viger::chat {
namespace {

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

std::string FallbackChatModel::generate(
    const std::vector<Message>& history,
    std::string_view user_message) {
    const std::string message = lower_ascii(std::string(user_message));
    if (message == "hello" || message == "hi" || message == "hey") {
        return "Hello! VIGER Chat session is working. Plug a local neural model into IChatModel next.";
    }
    if (message.find("help") != std::string::npos) {
        return "Chat foundation online. Commands: /clear, /history, /quit. A neural provider can replace this fallback.";
    }
    if (message.find("who are you") != std::string::npos) {
        return "I am the VIGER Chat runtime fallback. I keep conversation state, but I am not pretending to be a language model.";
    }
    return "I received: \"" + std::string(user_message) + "\" (history=" + std::to_string(history.size()) + ")";
}

ChatSession::ChatSession(std::unique_ptr<IChatModel> model)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("ChatSession requires a model provider");
    }
}

std::string ChatSession::send(std::string_view user_message) {
    if (user_message.empty()) {
        throw std::invalid_argument("Chat message cannot be empty");
    }

    history_.push_back({Role::User, std::string(user_message)});
    try {
        const std::string response = model_->generate(history_, user_message);
        history_.push_back({Role::Assistant, response});
        return response;
    } catch (...) {
        history_.pop_back();
        throw;
    }
}

void ChatSession::add_system_message(std::string message) {
    if (!message.empty()) {
        history_.push_back({Role::System, std::move(message)});
    }
}

void ChatSession::clear() {
    history_.clear();
}

} // namespace viger::chat
