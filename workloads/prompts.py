"""Shared prompt bank for benchmark clients.

We use 16 short, varied prompts so up to N=16 concurrent clients each get a
distinct one. Each is short enough that prefill is fast and decode dominates
the timing.
"""
PROMPTS = [
    "Write a short Python function that reverses a string.",
    "Explain in 3 sentences what Rust borrow checker does.",
    "Tell me a one-paragraph story about a robot learning to paint.",
    "Give 5 surprising facts about octopuses.",
    "What's the difference between TCP and UDP? Brief.",
    "Compose a 4-line poem about morning coffee.",
    "List 3 differences between SQL and NoSQL databases.",
    "How does a neural network learn? Explain like I'm 12.",
    "Describe a sunrise over the ocean in 50 words.",
    "List 4 use cases for Redis.",
    "What is gradient descent? One paragraph.",
    "Compare REST and gRPC briefly.",
    "Explain CAP theorem in 3 sentences.",
    "What's a binary search tree? Quick definition.",
    "Name 3 features of Kubernetes.",
    "Describe how DNS resolution works.",
]
