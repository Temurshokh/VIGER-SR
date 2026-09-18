"""Build a deterministic ~1M-token local English training corpus.

The generated text is original synthetic training material. It mixes:
- everyday dialogue and casual English;
- short narratives;
- science, history, technology, and school explanations;
- questions, answers, comparisons, cause/effect, uncertainty, and advice;
- varied vocabulary and sentence structures.

It is intentionally local and reproducible so VIGER does not need to download a text corpus.
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "english_1m.txt"
TARGET_BPE_TOKENS = 1_000_000

NAMES = [
    "Alex", "Maya", "Daniel", "Sara", "Tom", "Lina", "Amir", "Nora",
    "Leo", "Emma", "Jack", "Sofia", "Adam", "Mia", "Ethan", "Olivia",
    "Noah", "Ava", "Lucas", "Ella", "Ryan", "Zoe", "Ben", "Anna",
]
PLACES = [
    "school", "home", "the park", "the library", "the station", "the workshop",
    "the city center", "a small café", "the computer room", "the museum",
    "the laboratory", "the bus stop", "the sports field", "the market",
]
OBJECTS = [
    "a notebook", "a phone", "a laptop", "a camera", "a bicycle", "a book",
    "a backpack", "a key", "a map", "a small box", "a robot", "a game",
]
ADJECTIVES = [
    "quiet", "busy", "strange", "simple", "difficult", "useful", "interesting",
    "unexpected", "careful", "curious", "reliable", "practical", "complex",
    "ordinary", "temporary", "surprising", "familiar", "unusual", "ancient",
    "modern", "important", "tiny", "massive", "bright", "dark", "friendly",
]
VERBS = [
    "walked", "noticed", "found", "opened", "checked", "carried", "remembered",
    "explained", "decided", "changed", "tested", "built", "repaired", "shared",
    "compared", "discovered", "returned", "continued", "waited", "searched",
]
TOPICS = [
    "language learning", "computer programs", "space exploration", "history",
    "weather", "animals", "friendship", "science", "music", "games",
    "engineering", "school projects", "photography", "machines", "memory",
    "curiosity", "problem solving", "communication", "cities", "the internet",
]
CONNECTORS = [
    "because", "although", "however", "therefore", "meanwhile", "instead",
    "while", "unless", "even if", "as a result", "in addition", "for example",
]


def dialogue(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    friend = rng.choice([x for x in NAMES if x != name])
    replies = [
        f'{name}: Hey, what\'s up?\n{friend}: Not much. I\'m just working on something.\n{name}: What are you working on?\n{friend}: A small project. It is taking longer than I expected, but I\'m learning a lot.',
        f'{name}: You coming with us?\n{friend}: Yeah, give me a second.\n{name}: No rush.\n{friend}: Thanks. I just need to grab my bag.',
        f'{name}: Did you finish the task?\n{friend}: Almost. I have one part left.\n{name}: Need any help?\n{friend}: Maybe. Let me explain what is going wrong.',
        f'{name}: How was your day?\n{friend}: Pretty good, actually. I studied in the morning and worked on a project later.\n{name}: Sounds productive.\n{friend}: It was, although I was tired by the evening.',
        f'{name}: Want to play a game?\n{friend}: Sure. What do you want to play?\n{name}: Something simple.\n{friend}: Fair enough. Give me a minute.',
        f'{name}: Why did you change the plan?\n{friend}: The first approach was too complicated.\n{name}: So what are you doing instead?\n{friend}: I am trying a smaller experiment and checking the results carefully.',
        f'{name}: Are you sure?\n{friend}: Not completely. There is still some uncertainty.\n{name}: Then what should we do?\n{friend}: We should test it again before making a strong claim.',
        f'{name}: I do not understand this word.\n{friend}: Show me the sentence.\n{name}: Here it is.\n{friend}: The context suggests that it means something close to this.',
    ]
    return rng.choice(replies)


def story(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    adj = rng.choice(ADJECTIVES)
    verb = rng.choice(VERBS)
    return (
        f"{name} was walking near {place} when they noticed {obj}. "
        f"It looked {adj}, so they stopped and {verb} it carefully. "
        f"At first, they thought the object was ordinary. However, after a few minutes, "
        f"they realized that the situation was more interesting than it had seemed. "
        f"{name} decided not to rush. Instead, they observed the details, considered several "
        f"possible explanations, and wrote down what they knew. Later, a friend arrived and "
        f"asked what had happened. {name} explained the situation from the beginning. "
        f"The explanation was simple: one small detail had changed the whole story."
    )


def explanation(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    connector = rng.choice(CONNECTORS)
    pairs = [
        f"{topic.capitalize()} can be difficult at first because new ideas often appear unrelated. "
        f"{connector.capitalize()}, repeated examples help people notice patterns and build a clearer mental model.",
        f"When people study {topic}, they usually need both information and context. "
        f"A definition can explain one part of an idea, while an example shows how that idea works in practice.",
        f"A useful way to approach {topic} is to separate the problem into smaller questions. "
        f"First identify what is known, then identify what is uncertain, and finally decide what evidence would change the conclusion.",
        f"{topic.capitalize()} changes over time. What works in one situation may fail in another, "
        f"so a flexible approach is often more useful than a single rigid rule.",
    ]
    return rng.choice(pairs)


def science_or_history(rng: random.Random) -> str:
    passages = [
        "The scientific method uses observation, measurement, testing, and revision. A hypothesis is a proposed explanation that can be compared with evidence. If an experiment produces a result that conflicts with the hypothesis, the hypothesis may need to be revised. Repeating an experiment helps determine whether a result is stable or whether it may have been influenced by chance.",
        "The Earth has a dynamic climate system involving the atmosphere, oceans, land, ice, and living organisms. Weather describes short-term conditions, while climate describes patterns over longer periods. A single cold day does not by itself describe a climate trend. Scientists therefore examine measurements across longer time periods and use multiple sources of evidence.",
        "World War II was a global conflict from 1939 to 1945. Major events affected Europe, Asia, Africa, and the Pacific, while civilian populations experienced displacement, destruction, shortages, persecution, and mass death. Studying the war involves military history as well as political decisions, economies, societies, technology, and the experiences of ordinary people.",
        "The internet is a network of networks that allows computers and devices to exchange data using standardized protocols. A web page may travel through several systems before reaching a user. This is why a website can depend on many components even when the visible page looks simple.",
        "Astronomy studies objects and processes beyond Earth, including stars, planets, galaxies, and the evolution of the universe. Light carries information about distant objects. By analyzing that light, scientists can infer properties such as temperature, chemical composition, and motion.",
        "A computer stores information as data and processes it according to instructions. Modern software systems are built from many layers, including hardware, operating systems, libraries, applications, and network services. A small visible feature can therefore depend on a surprisingly large chain of components.",
    ]
    return rng.choice(passages)


def grammar_in_context(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    place = rng.choice(PLACES)
    templates = [
        f"{name} usually goes to {place} after school, but today they stayed home because they had an important task to finish.",
        f"If {name} has enough time tomorrow, they will visit {place} and look for a quieter place to study.",
        f"Although {name} had never used the tool before, they learned how it worked after reading the instructions and trying a small example.",
        f"{name} said that the project would be easier if the team divided the work into smaller parts.",
        f"The book that {name} borrowed last week was returned yesterday because another student needed it.",
        f"Having finished the experiment, {name} compared the results with the original prediction and noticed a small difference.",
        f"{name} might join the group later, provided that the earlier task is completed on time.",
        f"The problem was caused by a configuration that had been changed without anyone noticing.",
    ]
    return rng.choice(templates)


def generate(seed: int = 42, target_tokens: int = TARGET_BPE_TOKENS) -> Path:
    from python.viger_tiny_lm import TinyBPE

    rng = random.Random(seed)
    chunks: list[str] = []
    raw_chars = 0
    # Aim for more raw text than the final target because BPE compresses bytes.
    target_chars = int(target_tokens * 4.2)

    while raw_chars < target_chars:
        selector = rng.randrange(100)
        if selector < 28:
            text = dialogue(rng)
        elif selector < 53:
            text = story(rng)
        elif selector < 76:
            text = explanation(rng)
        elif selector < 90:
            text = grammar_in_context(rng)
        else:
            text = science_or_history(rng)
        chunks.append(text)
        raw_chars += len(text) + 2

    corpus = "\n\n".join(chunks)
    tokenizer = TinyBPE.train(corpus, max_merges=512)
    token_count = len(tokenizer.encode(corpus))

    # Top up in large batches if BPE compression was stronger than expected.
    while token_count < target_tokens:
        extra: list[str] = []
        for _ in range(3000):
            choice = rng.randrange(4)
            extra.append(
                dialogue(rng)
                if choice == 0
                else story(rng)
                if choice == 1
                else explanation(rng)
                if choice == 2
                else grammar_in_context(rng)
            )
        corpus += "\n\n" + "\n\n".join(extra)
        tokenizer = TinyBPE.train(corpus, max_merges=512)
        token_count = len(tokenizer.encode(corpus))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(corpus, encoding="utf-8")
    print(f"[VIGER Corpus] wrote {OUTPUT}")
    print(f"[VIGER Corpus] BPE tokens={token_count:,} (target={target_tokens:,})")
    print(f"[VIGER Corpus] characters={len(corpus):,}")
    return OUTPUT


def ensure_corpus(target_tokens: int = TARGET_BPE_TOKENS) -> Path:
    if OUTPUT.exists() and OUTPUT.stat().st_size > 2_000_000:
        return OUTPUT
    return generate(target_tokens=target_tokens)


if __name__ == "__main__":
    generate()
