#include "viger_sr/chat.hpp"

#include <cassert>
#include <memory>
#include <string>

int main() {
    using namespace viger::chat;

    ChatSession session(std::make_unique<FallbackChatModel>());
    session.add_system_message("test system");

    const std::string first = session.send("hello");
    assert(first.find("Hello") != std::string::npos);
    assert(session.size() == 3);

    session.clear();
    assert(session.size() == 0);

    return 0;
}
