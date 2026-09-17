#include "viger_sr/chat.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

int main() {
    using viger::chat::ChatSession;
    using viger::chat::FallbackChatModel;

    std::cout << "VIGER Chat 0.2\n"
              << "Local conversation runtime foundation\n"
              << "Type /help for commands.\n\n";

    ChatSession session(std::make_unique<FallbackChatModel>());
    session.add_system_message("You are VIGER, a concise local assistant.");

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (line == "/clear") {
            session.clear();
            std::cout << "[VIGER] conversation cleared\n";
            continue;
        }
        if (line == "/history") {
            std::cout << "[VIGER] messages in session: " << session.size() << "\n";
            continue;
        }
        if (line == "/help") {
            std::cout << "/clear  clear session\n"
                      << "/history show message count\n"
                      << "/quit   exit\n";
            continue;
        }
        if (line.empty()) {
            continue;
        }

        try {
            std::cout << "VIGER: " << session.send(line) << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[VIGER] error: " << error.what() << "\n";
        }
    }

    return EXIT_SUCCESS;
}
