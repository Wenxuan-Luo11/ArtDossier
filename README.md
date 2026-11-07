<div align="center">

# ArtDossier 🖼️🔍
### Multimodal RAG Engine for Grounded Cultural Analytics & Visual Interpretation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Vector Store: Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Framework: LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-purple.svg)](https://www.llamaindex.ai/)
[![VLM: Qwen3-VL](https://img.shields.io/badge/Models-Qwen3--VL%20%7C%20DeepSeek--V4-green.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An end-to-end vision-language pipeline that mitigates multimodal hallucination by decoupling blind visual perception from multi-source hybrid RAG retrieval and citation-grounded reasoning.*

[System Architecture](#-system-architecture) • [Key Features](#-key-features) • [Benchmark Results](#-benchmark--empirical-evaluation) • [Quickstart](#-quickstart) • [Methodology](#-research--methodology)

---

</div>

## 📌 Executive Summary

While contemporary Vision-Language Models (VLMs) can generate fluent surface-level descriptions, they suffer from two critical failure modes in domain-specific workflows: **visual hallucination** and an inability to bridge the **semantic gap** between raw pixels and nuanced contextual knowledge.

**ArtDossier** is a production-grade multimodal retrieval-augmented generation (RAG) framework developed to resolve this bottleneck. Given an unlabelled high-resolution image, the pipeline:
1. **Perceives without metadata bias:** Uses an 8B vision model to extract formal visual descriptors, objects, and iconographic queries *blind* to title and artist.
2. **Retrieves across authoritative corpora:** Executes hybrid vector search (BGE-M3 in Qdrant) + cross-encoder reranking (BGE-Reranker-v2-m3) across 6,557 ingested scholarly chunks (Rijksmuseum, Leiden Collection, Web Gallery of Art) alongside live web search.
3. **Reasoning & synthesis:** Employs a 32B thinking model that treats initial visual captions as a biased prior, audits retrieved evidence in an inspectable trace, and compiles an eight-layer domain dossier with strict inline source attribution.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph P1["Panel 1: Blind Visual Perception (Qwen3-VL-8B)"]
        A[Input Painting Image] --> B[Visual Feature Extraction]
        B --> C[3-Query Decomposition: Visual / Iconography / Synthesis]
    end

    subgraph RAG["Hybrid Knowledge Retrieval & Reranking"]
        C -->|Dense Vector Embedding BGE-M3| D[(Qdrant Vector DB<br/>6,557 Chunks / 3,425 Sources)]
        C -->|spaCy Noun Extraction| E[Reciprocal Rank Fusion Keyword Search]
        C -->|Synthesis Query| F[Live Web Context Engine]
        D & E --> G[BGE-Reranker-v2-m3 Cross-Encoder]
        G & F --> H[Ranked & Source-Prioritized Evidence Set]
    end

    subgraph P2["Panel 2: Auditable Stepwise Reasoning"]
        A & H --> I[Thinking Trace: Qwen3-VL-32B-Thinking]
        I --> J[Identify Discrepancies & Audit Mismatches]
    end

    subgraph P3["Panel 3: Structured 8-Layer Dossier Synthesis"]
        J --> K[Layer 1: Socio-Historical Context]
        J --> L[Layer 2-4: Artist / Materiality / Appearance]
        J --> M[Layer 5-6: Situated Symbols / Traditions]
        J --> N[Layer 7-8: Epistemic Horizons & Ambiguity]
        K & L & M & N --> O[Export Audit-Ready JSON / Dossier with Inline Citations]
    end

    style P1 fill:#f8f9fa,stroke:#0d6efd,stroke-width:2px
    style RAG fill:#f1f8ff,stroke:#0969da,stroke-width:2px
    style P2 fill:#fff8e6,stroke:#bf8700,stroke-width:2px
    style P3 fill:#f6ffed,stroke:#52c41a,stroke-width:2px
```

---

## ⚡ Key Engineering & Design Decisions

- **Inference-Time Metadata Blindness:** Unlike standard RAG setups that query by database ID or known titles, ArtDossier receives *zero* title or metadata prompts. It tests genuine computer-vision-to-text retrieval in zero-shot conditions.
- **Hierarchical Document Authority:** Prompts enforce strict epistemic tiers: Scholarly Catalogues (e.g., Leiden Collection) > Institutional Archives (Rijksmuseum API) > General Web Catalogues (WGA).
- **Inspectable Thinking Trace:** Panel 2 isolates the model's self-critique and alignment logic, preventing black-box confabulation before compiling the final client-facing dossier.
- **Dual Evaluation Regime:** Combines automated NLP metrics (F1, METEOR, BERTScore, CLIPScore) with parallel qualitative evaluations by domain humans and an independent LLM judge (`DeepSeek-V4-Flash`).
