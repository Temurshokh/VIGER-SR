"""Generate a larger, higher-diversity English corpus for VIGEROID 6.

The corpus is original synthetic material generated from varied templates. It
is intentionally structured around conversations, narratives, explanations,
reasoning, and factual prose instead of repetitive flash-card pairs.

The generated file stays local and is ignored by Git.
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "english_quality_2m.txt"
TARGET_CHARS = 8_500_000
SEED = 20260918

NAMES = [
    "Alex", "Maya", "Daniel", "Sara", "Tom", "Lina", "Amir", "Nora",
    "Leo", "Emma", "Jack", "Sofia", "Adam", "Mia", "Ethan", "Olivia",
    "Noah", "Ava", "Lucas", "Ella", "Ryan", "Zoe", "Ben", "Anna",
    "David", "Layla", "Omar", "Julia", "Sam", "Chris",
]

PLACES = [
    "school", "home", "the park", "the library", "the station",
    "the workshop", "the city center", "a small café", "the computer room",
    "the museum", "the laboratory", "the bus stop", "the sports field",
    "the market", "the old bridge", "the train platform", "the neighborhood",
]

OBJECTS = [
    "a notebook", "a phone", "a laptop", "a camera", "a bicycle", "a book",
    "a backpack", "a key", "a map", "a small box", "a robot", "a game",
    "a broken watch", "a folded letter", "a photograph", "a tool", "a ticket",
]

TOPICS = [
    "language learning", "computer programs", "space exploration", "history",
    "weather", "animals", "friendship", "science", "music", "games",
    "engineering", "school projects", "photography", "machines", "memory",
    "curiosity", "problem solving", "communication", "cities", "the internet",
    "climate", "mathematics", "art", "transportation", "education",
]

CONTEXTS = [
    "the project", "the assignment", "the experiment", "the trip",
    "the conversation", "the problem", "the plan", "the result",
    "the new version", "the meeting", "the test", "the story",
]

CASUAL_OPENERS = [
    "Hey, what's up?", "Yo, how's it going?", "Hey bro, you good?",
    "What's going on?", "How have you been?", "You free right now?",
    "Got a minute?", "Guess what?", "No way.", "Seriously?",
]

CASUAL_REPLIES = [
    "Not much. I'm just working on something.",
    "Yeah, I'm good. What's up with you?",
    "Pretty good, actually. I've been busy, though.",
    "Give me a second. I'm almost done.",
    "Honestly, I have no idea yet.",
    "Yeah, sure. What did you have in mind?",
    "I think so, but let me check first.",
    "Not really. I was planning to relax.",
    "That's fair. I can see why you'd think that.",
    "Haha, yeah. That happens more often than you'd expect.",
]

CONNECTORS = [
    "because", "although", "however", "therefore", "meanwhile", "instead",
    "while", "unless", "even if", "as a result", "in addition",
    "for example", "on the other hand", "that said", "in other words",
    "eventually", "at first", "afterwards", "otherwise",
]

FACTS = [
    "Light in vacuum travels at exactly 299,792,458 meters per second.",
    "Water is composed of two hydrogen atoms and one oxygen atom in each molecule.",
    "Earth orbits the Sun once in about 365.25 days.",
    "The Moon is Earth's natural satellite and is tidally locked to Earth.",
    "Jupiter is the largest planet in the Solar System.",
    "The Milky Way is the galaxy that contains the Solar System.",
    "A byte contains eight bits in modern computing systems.",
    "UTF-8 is a variable-length encoding for Unicode.",
    "A CPU executes instructions and performs arithmetic, logic, and control operations.",
    "A GPU is designed for highly parallel workloads and is widely used for graphics and machine learning.",
    "An algorithm is a finite procedure for solving a problem or performing a computation.",
    "A parameter is a numerical value that a machine-learning training process can optimize.",
    "A tokenizer converts text into a sequence of units represented numerically by a language model.",
    "A Transformer uses attention mechanisms to model relationships between positions in a sequence.",
    "A language model predicts probabilities for possible next tokens given a previous context.",
    "Backpropagation computes gradients through a neural network using the chain rule.",
    "Overfitting occurs when a model fits training data too closely and generalizes poorly to new examples.",
    "Validation data is held out from parameter updates and can be used to estimate generalization.",
    "World War II lasted from 1939 to 1945 and became a global conflict involving many countries.",
    "The Industrial Revolution involved major changes in manufacturing, energy use, transport, and urbanization.",
    "Natural selection can change the frequency of heritable traits in a population over generations.",
    "The human heart has four chambers: two atria and two ventricles.",
    "A molecule is a group of atoms held together by chemical bonds.",
    "An acid is a substance that can donate a proton in the Brønsted-Lowry model.",
    "Probability is commonly represented on a scale from 0 to 1 in mathematical models of uncertainty.",
]

def dialogue(rng: random.Random) -> str:
    a, b = rng.sample(NAMES, 2)
    opener = rng.choice(CASUAL_OPENERS)
    first = rng.choice(CASUAL_REPLIES)
    followups = [
        f"{a}: {opener}\n{b}: {first}\n{a}: What are you working on?\n{b}: I'm trying to fix {rng.choice(CONTEXTS)}. It looked simple at first, but there are a few details I didn't expect.",
        f"{a}: {opener}\n{b}: {first}\n{a}: Want to grab something to eat?\n{b}: Sure. Give me five minutes and I'll be ready.",
        f"{a}: Did you finish {rng.choice(CONTEXTS)}?\n{b}: Almost. I only have one part left.\n{a}: Need any help?\n{b}: Maybe. Let me show you what I've got so far.",
        f"{a}: I don't understand what this means.\n{b}: That's okay. Which part is confusing?\n{a}: The last sentence.\n{b}: Let's look at the context and work through it together.",
        f"{a}: Are you sure about that?\n{b}: Not completely. There is still some uncertainty.\n{a}: Then we should probably test it again.\n{b}: Agreed. I'd rather check than guess.",
        f"{a}: No way, you actually fixed it?\n{b}: Yeah! It turned out to be one tiny mistake.\n{a}: Classic. One line causes an hour of debugging.\n{b}: Exactly.",
        f"{a}: What's the plan?\n{b}: First we collect the information, then we compare the options.\n{a}: And after that?\n{b}: We choose the simplest solution that meets the requirements.",
    ]
    return rng.choice(followups)

def story(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    friend = rng.choice([x for x in NAMES if x != name])
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    return (
        f"{name} was walking near {place} when they noticed {obj}. "
        "At first, they assumed it was ordinary and kept walking. "
        "A few steps later, however, they stopped and looked back. "
        "Something about the object seemed familiar, although they could not immediately explain why. "
        f"They picked it up and noticed that {friend}'s name was written on the back. "
        f"{name} called {friend}, but the phone went unanswered. "
        "Instead of making a quick assumption, they decided to investigate carefully. "
        f"They returned to {place}, asked a few questions, and compared what they had learned. "
        "By the end of the afternoon, the mystery had a simple explanation, but the experience "
        "had taught them an important lesson: a first impression is useful, but it is not always correct."
    )

def reasoning(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    context = rng.choice(CONTEXTS)
    connector = rng.choice(CONNECTORS)
    return (
        f"Suppose a team is working on {topic}. The first version of {context} looks promising, "
        f"but one test produces an unexpected result. {connector.capitalize()}, the team should not "
        "immediately conclude that the entire idea is wrong. A better approach is to identify what "
        "changed, repeat the relevant test, and compare the new evidence with the original expectation. "
        "If the result remains stable, it becomes stronger evidence. If the result disappears, the team "
        "has learned that the original observation may have been caused by noise, measurement error, or "
        "an unusual condition. Good reasoning therefore depends not only on having information, but on "
        "using information carefully."
    )

def explanation(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    fact = rng.choice(FACTS)
    return (
        f"{topic.capitalize()} is easier to understand when ideas are connected. "
        f"For example, {fact} "
        "This does not mean that one fact explains the whole subject. "
        "A useful explanation separates the central idea from supporting details, "
        "defines unfamiliar terms, and gives an example that shows how the idea works in practice."
    )

def uncertainty(rng: random.Random) -> str:
    subject = rng.choice(CONTEXTS)
    options = [
        "The evidence suggests that the change helped, but it is too early to say how much it contributed.",
        "The result could be genuine, although another explanation is still possible.",
        "It seems likely that the problem is related to the recent change, but we should verify that with another test.",
        "I cannot be certain from the available information. More evidence would reduce the uncertainty.",
        "There are several plausible explanations, and the current evidence does not clearly distinguish between them.",
    ]
    return f"When evaluating {subject}, {rng.choice(options)}"

def instruction(rng: random.Random) -> str:
    task = rng.choice([
        "debug a program", "learn a difficult concept", "write a report",
        "prepare for an exam", "organize a project", "compare two options",
        "understand an unfamiliar word", "investigate an unexpected result",
    ])
    return (
        f"To {task}, start by stating the goal in one sentence. "
        "Then list the information you already have and separate it from what you still need to know. "
        "Break the task into smaller steps, complete the easiest useful step first, and check the result "
        "before moving on. If something fails, record what happened instead of changing many things at once."
    )

def generate(target_chars: int = TARGET_CHARS, seed: int = SEED) -> Path:
    rng = random.Random(seed)
    blocks: list[str] = []
    total = 0
    generators = [dialogue, story, reasoning, explanation, uncertainty, instruction]

    while total < target_chars:
        generator = rng.choice(generators)
        text = generator(rng)
        blocks.append(text)
        total += len(text) + 2

    corpus = "\n\n".join(blocks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(corpus, encoding="utf-8")
    print(f"[VIGER Corpus v2] wrote {OUTPUT}")
    print(f"[VIGER Corpus v2] chars={len(corpus):,}")
    print("[VIGER Corpus v2] corpus target is approximately 2M BPE tokens;")
    print("[VIGER Corpus v2] exact token count will be reported by the trainer.")
    return OUTPUT

def ensure_corpus() -> Path:
    if OUTPUT.exists() and OUTPUT.stat().st_size >= TARGET_CHARS * 0.95:
        return OUTPUT
    return generate()

if __name__ == "__main__":
    generate()
