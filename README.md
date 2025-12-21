# DAG-Based Semantic Intent Routing Engine

A graph-based, transparent NLP intent routing engine designed to handle complex user queries with overlapping intents, ambiguity, and multi-step context — without relying on brittle rule-based logic or large language models.

The system routes user input through a **DAG-structured intent graph** using semantic similarity, adaptive confidence thresholds, and a rolling execution memory, while exposing every routing decision, score, and fallback step for full transparency and debugging.

A live demo is deployed on **Hugging Face Spaces** with an API and web interface.

---

## High-Level Overview

This project answers a specific problem:

> *How can we reliably route natural language queries through a structured intent system when user input is ambiguous, incomplete, or contains multiple intents — without hiding logic behind black-box models?*

The engine treats intent resolution as a **graph traversal problem**, not a classification problem.  
Each user query is evaluated step-by-step through an intent DAG, with confidence-based routing and explicit fallback behavior.

Key design goals:
- Deterministic and explainable behavior
- Safe handling of ambiguity
- Support for multi-intent queries
- Clear separation between routing logic and deployment infrastructure

---

## Core Concepts

### Intent Graph (DAG)

- Intents are represented as nodes in a **Directed Acyclic Graph**
- Nodes may have multiple parents, allowing different user flows to converge
- Each node contains:
  - An intent label
  - Example utterances
  - Child intent nodes
  - Optional sibling-penalty behavior

The graph defines *what is possible*, not *what is chosen*.

---

### Semantic Routing with Embeddings

- User input is embedded using **SentenceTransformers (MiniLM)**
- Each node’s example utterances are pre-embedded and cached
- Routing decisions are made using **cosine similarity**
- At each step, the engine compares the query embedding against all child nodes

No hard rules. No keyword matching. No prompt logic.

---

### Dynamic Confidence Thresholds

Routing decisions are governed by adaptive thresholds:

- **High confidence threshold**
  - Required to confidently move forward in the graph
  - Decays with query length and traversal depth
- **Low confidence threshold**
  - Minimum signal required to avoid immediate failure

This prevents:
- Overconfidence on vague queries
- Dead ends deeper in the graph

---

### Sibling Penalization

Some intent nodes are semantically very close (e.g., “fix start date” vs “fix end date”).

For these cases:
- A sibling penalty is applied
- Scores are normalized so only the strongest sibling advances
- Prevents oscillation and accidental multi-routing

This allows fine-grained intent resolution without hard-coded rules.

---

## Multi-Intent Handling

The engine supports **multiple intents in a single user input**.

Flow:
1. Input is split into sub-queries using sentence and conjunction parsing
2. Each sub-query is routed independently
3. Competing intent paths are **executed one at a time**
4. Remaining paths are blocked until the current path completes
5. Context from completed paths is carried forward

This enables controlled, deterministic handling of compound requests like:

> “Book a flight for next week and cancel my trip in December.”

---

## Rolling Execution Memory

The engine maintains a **short-horizon rolling memory** of recent intent states:

- Implemented as a fixed-size deque
- Stores the most recent intent labels
- Updated after each successful routing step
- Used only during ambiguity or low-confidence situations

Purpose:
- Bias routing toward recently active intent paths
- Maintain conversational coherence
- Reduce repeated clarification prompts

This is **execution-state memory**, not retrieval or persistence.

---

## Fallback and Ambiguity Handling

When the engine cannot confidently route a query, it does **not guess**.

Fallback behavior includes:

### 1. Ambiguity Zone Handling
When confidence is between low and high thresholds:
- The parent node remains active
- Rolling execution memory is used to bias routing
- If ambiguity remains, the engine asks for clarification

### 2. Backtracking
When confidence drops below the low threshold:
- The engine backtracks to the previous node
- Attempts re-routing with updated context
- Prevents false or unsafe intent execution

### 3. Blocking Clarification
Clarification is **blocking**, not advisory:
- Execution pauses
- User input is required
- Routing resumes with the new information

---

## Full Transparency by Design

Every traversal step is **explicit and visible**.

The engine exposes:
- Current node
- Candidate child intents
- Confidence scores
- Threshold values
- Routing decisions
- Backtracking events
- Fallback triggers

There is no hidden state or silent decision-making.

This makes the system:
- Easy to debug
- Easy to reason about
- Easy to extend safely

---

## NLP Preprocessing (Lexer-Style)

Before routing, input is normalized using lightweight NLP:

- **spaCy** is used to detect entities (DATE, PLACE, etc.)
- Detected entities are replaced with abstract tokens
- Original values are preserved separately

This acts like a **lexer**, not a parser:
- It simplifies semantic matching
- It does not complete or execute intents
- It keeps routing logic clean and domain-agnostic

---

## Architecture Overview

```

User Input
↓
Sentence / Clause Split (NLTK)
↓
Entity Normalization (spaCy)
↓
Embedding Generation (SentenceTransformers)
↓
Graph Traversal with Confidence Scoring
↓
Fallback / Clarification / Backtracking (if needed)
↓
Transparent Routing Output

```

---

## Deployment

The engine is deployed as a live service:

- **FastAPI** exposes the routing API
- **Docker** is used for containerization
- **React + Vite** provide a simple web interface
- Hosted on **Hugging Face Spaces**

The frontend exists to interact with the engine visually and does not affect routing logic.

---

## Performance

- ~91% intent routing accuracy (domain-specific evaluation)
- ~100ms average CPU-only inference latency
- Cached embeddings for fast traversal
- Deterministic execution with no external dependencies

---

## Tech Stack

- Python
- SentenceTransformers (MiniLM)
- PyTorch
- spaCy
- NLTK
- NumPy
- FastAPI
- Docker
- React
- Vite
- Hugging Face Spaces

---

## Non-Goals

This project intentionally does **not** include:
- Large Language Models
- Prompt-based logic
- Persistent memory or retrieval systems
- Intent execution or action fulfillment

The focus is **intent routing**, not automation.

---

## Summary

This engine demonstrates how structured graphs, semantic similarity, and careful confidence control can replace fragile rule-based systems for intent routing - while remaining transparent, debuggable, and fast.
