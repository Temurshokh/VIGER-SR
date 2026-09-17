#include "viger_sr/chat.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace {

void print_help() {
    std::cout << "VIGER Chat 0.3\n"
              << "Local conversation runtime\n\n"
              << "Modes:\n"
              << "  viger-chat              built-in fallback\n"
              << "  viger-chat --local     connect to local model at 127.0.0.1:8080\n"
              << "  viger-chat --local HOST PORT\n\n"
              << "Commands:\n"
              << "  /clear     clear conversation\n"
              << "  /history   show message count\n"
              << "  /help      show commands\n"
              << "  /quit      exit\n\n";
}

} // namespace

int main(int argc, char** argv) {
    using namespace viger::chat;

    bool local = false;
    LocalHttpChatModel::Config local_config;

    if (argc > 1) {
        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            print_help();
            return EXIT_SUCCESS;
        }
        if (std::string(argv[1]) == "--local") {
            local = true;
            if (argc >= 3) local_config.host = argv[2];
            if (argc >= 4) local_config.port = static_cast<std::uint16_t>(std::stoul(argv[3]));
            if (argc > 4) {
                print_help();
                return EXIT_FAILURE;
            }
        } else {
            print_help();
            return EXIT_FAILURE;
        }
    }

    std::cout << "VIGER Chat 0.3\n"
              << (local ? "Provider: local HTTP model\n" : "Provider: built-in fallback\n")
              << "Type /help for commands.\n\n";

    std::unique_ptr<IChatModel> model;
    if (local) {
        model = std::make_unique<LocalHttpChatModel>(local_config);
    } else {
        model = std::make_unique<FallbackChatModel>();
    }

    ChatSession session(std::move(model));
    session.add_system_message(
        "You are VIGER, a concise local assistant. Answer clearly and avoid inventing facts.");

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "/quit" || line == "/exit") break;
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
            print_help();
            continue;
        }
        if (line.empty()) continue;

        try {
            std::cout << "VIGER: " << session.send(line) << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[VIGER] error: " << error.what() << "\n";
            if (local) {
                std::cerr << "[VIGER] Is the local model runtime running on "
                          << local_config.host << ':' << local_config.port << "?\n";
            }
        }
    }

    return EXIT_SUCCESS;
}
