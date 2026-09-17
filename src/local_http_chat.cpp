#include "viger_sr/chat.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
constexpr socket_t invalid_socket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t invalid_socket = -1;
#endif

namespace viger::chat {
namespace {

class SocketRuntime {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data{};
        const int result = WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            throw std::runtime_error("WSAStartup failed: " + std::to_string(result));
        }
        active_ = true;
#endif
    }

    ~SocketRuntime() {
#ifdef _WIN32
        if (active_) {
            WSACleanup();
        }
#endif
    }

private:
#ifdef _WIN32
    bool active_ = false;
#endif
};

class SocketHandle {
public:
    explicit SocketHandle(socket_t socket) : socket_(socket) {}
    ~SocketHandle() {
        if (socket_ != invalid_socket) {
#ifdef _WIN32
            closesocket(socket_);
#else
            close(socket_);
#endif
        }
    }

    SocketHandle(const SocketHandle&) = delete;
    SocketHandle& operator=(const SocketHandle&) = delete;

    socket_t get() const noexcept { return socket_; }

private:
    socket_t socket_;
};

void set_timeout(socket_t socket, int timeout_ms) {
#ifdef _WIN32
    const DWORD timeout = static_cast<DWORD>(std::max(1, timeout_ms));
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    timeval timeout{};
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

std::string json_escape(std::string_view input) {
    std::string output;
    output.reserve(input.size() + 16);
    for (unsigned char c : input) {
        switch (c) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (c < 0x20) {
                char buffer[7]{};
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(c));
                output += buffer;
            } else {
                output.push_back(static_cast<char>(c));
            }
        }
    }
    return output;
}

void append_utf8(std::string& output, std::uint32_t cp) {
    if (cp <= 0x7F) {
        output.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        output.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::uint32_t hex4(std::string_view value) {
    if (value.size() < 4) {
        throw std::runtime_error("Invalid JSON unicode escape");
    }
    std::uint32_t result = 0;
    for (char c : value.substr(0, 4)) {
        result <<= 4;
        if (c >= '0' && c <= '9') result |= static_cast<std::uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') result |= static_cast<std::uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') result |= static_cast<std::uint32_t>(c - 'A' + 10);
        else throw std::runtime_error("Invalid JSON unicode escape");
    }
    return result;
}

std::string json_unescape_string(std::string_view encoded) {
    std::string output;
    output.reserve(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        const char c = encoded[i];
        if (c != '\\') {
            output.push_back(c);
            continue;
        }
        if (++i >= encoded.size()) throw std::runtime_error("Invalid JSON escape");
        switch (encoded[i]) {
        case '"': output.push_back('"'); break;
        case '\\': output.push_back('\\'); break;
        case '/': output.push_back('/'); break;
        case 'b': output.push_back('\b'); break;
        case 'f': output.push_back('\f'); break;
        case 'n': output.push_back('\n'); break;
        case 'r': output.push_back('\r'); break;
        case 't': output.push_back('\t'); break;
        case 'u': {
            if (i + 4 >= encoded.size()) throw std::runtime_error("Invalid JSON unicode escape");
            const std::uint32_t first = hex4(encoded.substr(i + 1, 4));
            i += 4;
            if (first >= 0xD800 && first <= 0xDBFF && i + 6 < encoded.size() && encoded[i + 1] == '\\' && encoded[i + 2] == 'u') {
                const std::uint32_t second = hex4(encoded.substr(i + 3, 4));
                if (second >= 0xDC00 && second <= 0xDFFF) {
                    const std::uint32_t cp = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                    append_utf8(output, cp);
                    i += 6;
                    break;
                }
            }
            append_utf8(output, first);
            break;
        }
        default:
            throw std::runtime_error("Unsupported JSON escape sequence");
        }
    }
    return output;
}

std::string extract_json_string(std::string_view json, std::string_view key, std::size_t start_at = 0) {
    const std::string quoted_key = "\"" + std::string(key) + "\"";
    const std::size_t key_pos = json.find(quoted_key, start_at);
    if (key_pos == std::string_view::npos) {
        throw std::runtime_error("JSON response has no '" + std::string(key) + "' field");
    }

    std::size_t pos = key_pos + quoted_key.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != ':') throw std::runtime_error("Malformed JSON field");
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') throw std::runtime_error("JSON field is not a string");
    ++pos;

    const std::size_t value_start = pos;
    bool escaped = false;
    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return json_unescape_string(json.substr(value_start, pos - value_start));
        }
    }
    throw std::runtime_error("Unterminated JSON string");
}

std::string role_name(Role role) {
    switch (role) {
    case Role::System: return "system";
    case Role::User: return "user";
    case Role::Assistant: return "assistant";
    }
    return "user";
}

std::string build_messages_json(const std::vector<Message>& history, std::size_t max_chars) {
    if (history.empty()) throw std::runtime_error("Cannot call local model with empty history");

    std::vector<std::size_t> selected;
    selected.reserve(history.size());
    std::size_t used = 0;

    for (std::size_t index = history.size(); index-- > 0;) {
        const std::size_t cost = history[index].content.size() + 32;
        if (!selected.empty() && used + cost > max_chars) break;
        selected.push_back(index);
        used += cost;
    }
    std::reverse(selected.begin(), selected.end());

    // Keep the first system message when context trimming removed it.
    if (selected.front() != 0 && history[0].role == Role::System) {
        selected.insert(selected.begin(), 0);
    }

    std::string json = "[";
    for (std::size_t n = 0; n < selected.size(); ++n) {
        if (n > 0) json += ",";
        const Message& message = history[selected[n]];
        json += "{\"role\":\"" + role_name(message.role) + "\",\"content\":\"";
        json += json_escape(message.content);
        json += "\"}";
    }
    json += "]";
    return json;
}

std::string http_post(const LocalHttpChatModel::Config& config, const std::string& body) {
    if (config.host.empty() || config.endpoint.empty()) {
        throw std::invalid_argument("Local model host and endpoint cannot be empty");
    }
    if (config.timeout_ms <= 0) throw std::invalid_argument("Local model timeout must be positive");

    SocketRuntime runtime;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* addresses = nullptr;
    const std::string port = std::to_string(config.port);
    const int resolve_result = getaddrinfo(config.host.c_str(), port.c_str(), &hints, &addresses);
    if (resolve_result != 0 || addresses == nullptr) {
#ifdef _WIN32
        throw std::runtime_error("Could not resolve local model host (WSA=" + std::to_string(resolve_result) + ")");
#else
        throw std::runtime_error("Could not resolve local model host: " + std::string(gai_strerror(resolve_result)));
#endif
    }

    std::string response;
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        SocketHandle socket(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (socket.get() == invalid_socket) continue;
        set_timeout(socket.get(), config.timeout_ms);
        if (::connect(socket.get(), address->ai_addr, static_cast<int>(address->ai_addrlen)) != 0) continue;

        std::ostringstream request;
        request << "POST " << config.endpoint << " HTTP/1.1\r\n"
                << "Host: " << config.host << ':' << config.port << "\r\n"
                << "Content-Type: application/json\r\n"
                << "Accept: application/json\r\n"
                << "Connection: close\r\n";
        if (!config.api_key.empty()) request << "Authorization: Bearer " << config.api_key << "\r\n";
        request << "Content-Length: " << body.size() << "\r\n\r\n" << body;
        const std::string raw = request.str();

        std::size_t sent = 0;
        while (sent < raw.size()) {
#ifdef _WIN32
            const int written = ::send(socket.get(), raw.data() + sent, static_cast<int>(raw.size() - sent), 0);
#else
            const ssize_t written = ::send(socket.get(), raw.data() + sent, raw.size() - sent, 0);
#endif
            if (written <= 0) break;
            sent += static_cast<std::size_t>(written);
        }
        if (sent != raw.size()) continue;

        std::array<char, 8192> buffer{};
        for (;;) {
#ifdef _WIN32
            const int received = ::recv(socket.get(), buffer.data(), static_cast<int>(buffer.size()), 0);
#else
            const ssize_t received = ::recv(socket.get(), buffer.data(), buffer.size(), 0);
#endif
            if (received <= 0) break;
            response.append(buffer.data(), static_cast<std::size_t>(received));
        }
        break;
    }
    freeaddrinfo(addresses);

    if (response.empty()) {
        throw std::runtime_error("Could not connect to local model at " + config.host + ':' + std::to_string(config.port));
    }

    const std::size_t status_begin = response.find(' ');
    const std::size_t status_end = response.find(' ', status_begin + 1);
    if (status_begin == std::string::npos || status_end == std::string::npos) {
        throw std::runtime_error("Malformed HTTP response from local model");
    }

    int status_code = 0;
    const auto status_text = std::string_view(response).substr(status_begin + 1, status_end - status_begin - 1);
    const auto [ptr, ec] = std::from_chars(status_text.data(), status_text.data() + status_text.size(), status_code);
    if (ec != std::errc{} || ptr != status_text.data() + status_text.size()) {
        throw std::runtime_error("Invalid HTTP status from local model");
    }

    const std::size_t body_begin = response.find("\r\n\r\n");
    const std::string response_body = body_begin == std::string::npos ? "" : response.substr(body_begin + 4);
    if (status_code < 200 || status_code >= 300) {
        std::string detail = response_body.empty() ? "no response body" : response_body.substr(0, 800);
        throw std::runtime_error("Local model HTTP " + std::to_string(status_code) + ": " + detail);
    }
    return response_body;
}

} // namespace

LocalHttpChatModel::LocalHttpChatModel(Config config)
    : config_(std::move(config)) {}

std::string LocalHttpChatModel::generate(
    const std::vector<Message>& history,
    std::string_view user_message) {
    if (history.empty()) throw std::invalid_argument("Local model requires conversation history");
    if (user_message.empty()) throw std::invalid_argument("Local model user message cannot be empty");
    if (config_.max_tokens <= 0) throw std::invalid_argument("Local model max_tokens must be positive");
    if (config_.temperature < 0.0f || config_.temperature > 2.0f) {
        throw std::invalid_argument("Local model temperature must be between 0 and 2");
    }

    const std::string messages = build_messages_json(history, config_.max_history_chars);
    std::ostringstream body;
    body << "{\"model\":\"" << json_escape(config_.model)
         << "\",\"messages\":" << messages
         << ",\"temperature\":" << config_.temperature
         << ",\"max_tokens\":" << config_.max_tokens
         << ",\"stream\":false}";

    const std::string response = http_post(config_, body.str());
    const std::size_t choices_pos = response.find("\"choices\"");
    if (choices_pos == std::string::npos) {
        throw std::runtime_error("Local model response contains no choices field");
    }
    const std::size_t message_pos = response.find("\"message\"", choices_pos);
    if (message_pos == std::string::npos) {
        throw std::runtime_error("Local model response contains no message field");
    }
    return extract_json_string(response, "content", message_pos);
}

} // namespace viger::chat
