# VIGER Chat provider contract

The chat runtime is intentionally split into two layers:

```text
UI / CLI
   |
ChatSession
   |
IChatModel
   |
+-------------------------------+
|                               |
FallbackChatModel      LocalHttpChatModel
                              |
                              v
                    local model runtime
```

`ChatSession` owns conversation state. A model provider receives the current history and the newest user message, then returns one assistant message.

## Why this boundary exists

The project should not care whether the model is:

- a tiny custom Transformer;
- a GGUF model through a native runtime;
- a local inference service on `127.0.0.1`;
- or a future C++ neural engine.

Only the provider changes. The session, commands, history, and future desktop UI remain the same.

## Local model adapter

`LocalHttpChatModel` implements the provider contract using an ordinary HTTP client written in the VIGER C++ source tree. It does not require libcurl or a JSON package.

The default endpoint is:

```text
http://127.0.0.1:8080/v1/chat/completions
```

The adapter sends the conversation as `system`, `user`, and `assistant` messages and parses the returned assistant `content`. It also trims old history by a configurable character budget before the request.

A compatible local runtime can be run separately on the same machine. For example, llama.cpp's server exposes an OpenAI-compatible `/v1/chat/completions` route and accepts chat `messages` plus generation parameters.

Example local workflow:

```text
GGUF model
   |
llama-server :8080
   |
HTTP /v1/chat/completions
   |
LocalHttpChatModel
   |
ChatSession
   |
viger-chat
```

Then start VIGER with:

```bash
viger-chat --local
```

or:

```bash
viger-chat --local 127.0.0.1 8080
```

The server path is preferred to launching a fresh model process for every message because one runtime can keep the conversation context alive between requests.

## Provider requirements

Implement:

```cpp
class IChatModel {
public:
    virtual ~IChatModel() = default;
    virtual std::string generate(
        const std::vector<Message>& history,
        std::string_view user_message) = 0;
};
```

A provider should:

1. preserve the order of system/user/assistant messages;
2. enforce a context limit before inference;
3. report model/runtime errors instead of returning fabricated text;
4. return UTF-8 text;
5. avoid writing model data into source-controlled files.

## Tiny custom model path

`python/train_tiny_chat.py` remains a research sandbox for a small byte-level Transformer. `python/run_tiny_chat.py` can load its checkpoint and generate text locally. It is intentionally separate from the production provider adapter so a better model can replace it without touching the conversation layer.

## Future providers

A true native GGUF/libllama provider can be added later when we want to remove the local HTTP process entirely. That change should implement `IChatModel`, not rewrite `ChatSession`.
