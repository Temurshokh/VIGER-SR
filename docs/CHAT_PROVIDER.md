# VIGER Chat provider contract

The chat runtime is intentionally split into two layers:

```text
UI / CLI
   |
ChatSession
   |
IChatModel
   |
+--------------------------+
|                          |
FallbackChatModel      neural provider
```

`ChatSession` owns conversation state. A model provider receives the current history and the newest user message, then returns one assistant message.

## Why this boundary exists

The project should not care whether the model is:

- a tiny custom Transformer;
- a GGUF model through a native runtime;
- a local inference service on `127.0.0.1`;
- or a future C++ neural engine.

Only the provider changes. The session, commands, history, and future desktop UI remain the same.

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

## Planned providers

### Tiny native model

A small custom causal Transformer can eventually use a compact tokenizer and memory-mapped weights. This is the most self-contained option, but it needs a real training corpus and a dedicated inference implementation.

### GGUF/native runtime adapter

A local GGUF model can be connected without changing `ChatSession`. The adapter owns tokenizer setup, prompt formatting, sampling, and context management.

### Local HTTP adapter

A local OpenAI-compatible inference server can also be wrapped behind `IChatModel`. This keeps VIGER itself free of a remote backend while allowing experimentation with different local models.
