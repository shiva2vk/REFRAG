# REFRAG: Retrieval-Enhanced Fragment-based Generation

## Overview

This repository contains two implementations of **REFRAG (Retrieval-Enhanced Fragment-based Generation)**, a novel approach to context compression and retrieval-augmented generation (RAG) using large language models. Instead of concatenating raw text passages, REFRAG compresses context into compact KV (Key-Value) prefixes that can be efficiently injected into the transformer's attention mechanism.

## 🎯 What is REFRAG?

Traditional RAG systems face a critical trade-off:
- **More context** = Better answers but exponentially higher compute costs
- **Less context** = Faster but potentially missing crucial information

**REFRAG solves this** by:
1. **Compressing** long passages into learned KV prefix representations
2. **Injecting** these compressed memories into the transformer's attention layers
3. **Selectively expanding** important chunks during inference based on attention patterns

This enables models to "remember" vast amounts of context while maintaining efficient token budgets.

---

## 📂 Repository Structure

```
refrag/
├── refrag_qLORA.ipynb              # QLoRA implementation (4-bit + LoRA adapters)
├── refrag_full_precision.ipynb     # Full precision bf16 implementation
├── README.md                       # This file
└── META_REFRAG.md                 # Detailed explanation of the meta-learning approach
```

---

## 🚀 Two Implementations

### 1. **QLoRA Version** (`refrag_qLORA.ipynb`)
- **Training**: 4-bit quantization with LoRA adapters
- **Memory**: ~6-8 GB VRAM for LLaMA-3.1-8B
- **Speed**: Faster training, smaller checkpoint sizes
- **Use Case**: Resource-constrained environments, rapid prototyping

### 2. **Full Precision Version** (`refrag_full_precision.ipynb`)
- **Training**: BFloat16 full model fine-tuning
- **Memory**: ~40-60 GB VRAM for LLaMA-3.1-8B
- **Speed**: Slower but potentially better quality
- **Use Case**: Production deployments, maximum performance

Both implementations follow the same 4-stage pipeline:
1. **Reconstruction** → Learn to compress
2. **CPT (Continual Pre-Training)** → Adapt model to compressed context
3. **SFT (Supervised Fine-Tuning)** → Task-specific fine-tuning
4. **Inference** → Compressed memory + selective expansion

---

## 🧠 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     REFRAG Pipeline                          │
└─────────────────────────────────────────────────────────────┘

1. ENCODING PHASE
   Documents → Chunks → RoBERTa Encoder → Embeddings (768-dim)
                                              ↓
2. COMPRESSION PHASE
   Embeddings → KV Projector → Compressed KV Prefixes
                                (per layer, per chunk)
                                              ↓
3. GENERATION PHASE
   Query + KV Prefixes → LLaMA Decoder → Response
              ↓
   (Optional) Attention-Guided Expansion
              ↓
   High-attention chunks → Inject raw text → Continue generation
```

### Key Components

#### **1. ChunkEncoder (RoBERTa)**
```python
class ChunkEncoder(nn.Module):
    """Encodes text chunks into dense embeddings"""
    - Input: Raw text chunks
    - Output: 768-dim embeddings (mean-pooled)
```

#### **2. KVProjector**
```python
class KVProjector(nn.Module):
    """Projects embeddings into per-layer KV prefixes"""
    - Input: 768-dim embeddings
    - Output: K, V tensors for each transformer layer
    - Architecture: MLP per layer (4x width expansion)
```

#### **3. AttnThresholdPolicy**
```python
class AttnThresholdPolicy:
    """Selectively expands high-attention chunks"""
    - Monitors attention weights over compressed prefixes
    - Selects top-k chunks based on attention scores
    - Injects raw text for selected chunks
```

---

## 📊 Training Stages

### Stage 1: Reconstruction (Projector Warm-up)
**Goal**: Teach the projector to compress text without losing information

```python
# Freeze LLM, train only projector
for chunk_text in corpus:
    embedding = encoder(chunk_text)
    K, V = projector(embedding)  # Learn compression
    loss = LLM.reconstruct(chunk_text, past_kv=(K,V))
```

**Outcome**: Projector learns to create KV prefixes that help the LLM reconstruct original text

---

### Stage 2: CPT (Continual Pre-Training)
**Goal**: Adapt the LLM to work with compressed context

```python
# Train both projector + LLM
for chunk_text in corpus:
    embedding = encoder(chunk_text)
    K, V = projector(embedding)

    # Randomly mask labels to encourage compression usage
    if random() < 0.8:
        labels = randomly_mask(labels)

    loss = LLM(chunk_text, past_kv=(K,V), labels=labels)
```

**Key Innovation**: Compression mixing - randomly mask 50% of tokens in labels
- Forces model to rely on compressed KV prefixes
- Teaches model to "decompress" information from prefixes

**Outcome**: LLM learns to extract information from compressed representations

---

### Stage 3: SFT (Supervised Fine-Tuning)
**Goal**: Fine-tune on task-specific instruction-response pairs

```python
# Standard instruction fine-tuning
for (instruction, response) in sft_data:
    prompt = f"### Instruction\n{instruction}\n\n### Response\n{response}"
    loss = LLM(prompt)  # No compressed context here
```

**Outcome**: Model learns task-specific behavior (e.g., answering network diagnostics questions)

---

### Stage 4: Inference with Selective Expansion
**Goal**: Answer queries using compressed memory + smart expansion

```python
# Compress all passages
for passage in knowledge_base:
    chunks = split_into_chunks(passage)
    embeddings = encoder(chunks)
    K, V = projector(embeddings)  # Compress entire corpus

# Generate with compressed memory
prompt = "Question: {query}\nAnswer:"
for step in range(max_tokens):
    output = LLM(prompt, past_kv=(K,V), output_attentions=True)
    next_token = argmax(output.logits)

    # Selective expansion at steps 32, 96
    if step in (32, 96):
        attn_scores = output.attentions[-1][:, :, -1, :prefix_length]
        top_chunks = select_top_k(attn_scores)
        expanded_text = "\n".join([chunks[i] for i in top_chunks])
        prompt += expanded_text  # Inject raw text
```

**Key Innovation**: Attention-guided expansion
- Monitor which compressed chunks get high attention
- Selectively inject raw text for those chunks
- Maintains efficiency while ensuring critical info is available

---

## 🔧 Configuration

Both notebooks use a centralized config dictionary:

```python
cfg = {
    # Models
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "encoder_id": "roberta-base",

    # Data paths
    "corpus_path": "/data/cpt_corpus.jsonl",     # CPT corpus
    "sft_path": "/data/sft.jsonl",               # SFT dataset
    "passages_path": "/data/passages.jsonl",     # Inference passages

    # Training hyperparams
    "prefix_len": 1,          # Tokens per compressed chunk
    "chunk_size": 16,         # Tokens per chunk
    "max_len": 4096,          # Max sequence length
    "epochs_recon": 1,
    "epochs_cpt": 1,
    "epochs_sft": 1,
    "lr_recon": 2e-4,
    "lr_cpt": 1e-4,
    "lr_sft": 5e-5,

    # Inference
    "query": "Your question here",
    "max_new_tokens": 256,
    "expand_budget": 256,     # Max tokens for expansion
}
```

---

## 📝 Data Formats

### CPT/Reconstruction Corpus (`corpus_path`)
```jsonl
{"id": "doc_001", "text": "BGP configuration for AS65001..."}
{"id": "doc_002", "text": "IDS alert patterns indicate..."}
```

### SFT Dataset (`sft_path`)
**Format 1**: inputs/targets
```jsonl
{"inputs": "Diagnose this BGP issue", "targets": "The root cause is..."}
```

**Format 2**: instruction/input/output
```jsonl
{"instruction": "Analyze network logs", "input": "...", "output": "..."}
```

### Inference Passages (`passages_path`)
```jsonl
{"id": "passage_001", "text": "Networking documentation..."}
{"id": "passage_002", "text": "Security configuration..."}
```

---

## 🎯 Use Cases

### 1. **Network/Security Diagnostics**
- Compress: Configuration files, logs, documentation
- Query: "Why is BGP flapping on router X?"
- Benefit: Model can "remember" entire config library

### 2. **Long Document QA**
- Compress: Research papers, legal documents, manuals
- Query: "What are the key findings?"
- Benefit: Process documents far exceeding context window

### 3. **Code Understanding**
- Compress: Large codebases, API docs
- Query: "How does authentication work?"
- Benefit: Navigate massive codebases efficiently

### 4. **Medical/Scientific Literature**
- Compress: PubMed abstracts, clinical notes
- Query: "What treatments are mentioned?"
- Benefit: Synthesize information across many documents

---

## 📈 Performance Characteristics

| Metric | Traditional RAG | REFRAG (Compressed) |
|--------|----------------|---------------------|
| Context Capacity | ~4K tokens | ~100K+ tokens (compressed) |
| Latency | Linear O(n) | Constant O(1) + selective O(k) |
| Memory | High | Low (KV prefixes only) |
| Accuracy | Good (if context fits) | Good (even with massive context) |
| Flexibility | Static retrieval | Dynamic expansion |

### Compression Ratio Example
```
Original: 10,000 chunks × 16 tokens = 160,000 tokens
Compressed: 10,000 chunks × 1 prefix = 10,000 "virtual tokens"
Ratio: 16:1 compression

Selective expansion: 256 tokens budget
→ ~16 most important chunks expanded
→ Total tokens: 10,000 (compressed) + 256 (expanded) ≈ 10K effective
```

---

## 🛠️ Setup & Usage

### Installation
```bash
# Install dependencies
pip install torch transformers accelerate datasets peft bitsandbytes faiss-cpu tqdm

# QLoRA version only
pip install bitsandbytes

# Download notebooks
git clone https://github.com/yourusername/refrag
cd refrag
```

### Running Training

#### QLoRA Version
```python
# 1. Edit config in notebook
cfg["corpus_path"] = "path/to/your/corpus.jsonl"
cfg["sft_path"] = "path/to/your/sft.jsonl"

# 2. Run stages sequentially
run_reconstruction(cfg)  # Stage 1
run_cpt(cfg)             # Stage 2
run_sft(cfg)             # Stage 3
```

#### Full Precision Version
```python
# Same as QLoRA but requires more VRAM
run_reconstruction(cfg)
run_cpt(cfg)
run_sft(cfg)
```

### Running Inference
```python
cfg["passages_path"] = "path/to/knowledge_base.jsonl"
cfg["query"] = "Your question here"
cfg["expand_budget"] = 256

run_infer(cfg)
```

---

## 🧪 Hyperparameter Tuning Tips

### For Better Compression
- Increase `prefix_len` (1 → 2 or 4): More capacity per chunk
- Increase projector width (`width_mult=4` → `8`): More expressiveness
- Train longer reconstruction (`epochs_recon=1` → `3`)

### For Better Accuracy
- Increase `expand_budget` (256 → 512): More raw text during inference
- Decrease `chunk_size` (16 → 8): Finer-grained chunks
- Train longer CPT (`epochs_cpt=1` → `3`)

### For Faster Training
- Use QLoRA version instead of full precision
- Decrease batch size if OOM (`bs=8` → `4`)
- Use smaller encoder (`roberta-base` → `distilroberta-base`)

---

## 🔬 Advanced Topics

### Custom Expansion Strategy
```python
# Default: expand at steps 32, 96
if step in (32, 96):
    expand_chunks()

# Custom: expand when perplexity spikes
if current_perplexity > threshold:
    expand_chunks()

# Custom: expand adaptively
if attention_entropy < threshold:
    expand_chunks()  # Model is uncertain
```

### Multi-Stage Compression
```python
# Level 1: 16 tokens → 1 prefix
# Level 2: 4 prefixes → 1 super-prefix
# Hierarchical compression for extreme contexts
```

### Domain Adaptation
```python
# CPT on domain corpus (e.g., medical)
run_cpt(cfg, corpus="medical_papers.jsonl")

# SFT on domain tasks
run_sft(cfg, sft="medical_qa.jsonl")
```

---

## 📚 References & Related Work

- **Perceiver**: [Jaegle et al., 2021](https://arxiv.org/abs/2103.03206) - Cross-attention with learned queries
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) - Low-rank adaptation
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) - Quantized LoRA
- **RETRO**: [Borgeaud et al., 2022](https://arxiv.org/abs/2112.04426) - Retrieval-enhanced transformers
- **AutoCompressors**: [Chevalier et al., 2023](https://arxiv.org/abs/2305.14788) - Learned compression

---

## 🤝 Contributing

This is a research implementation. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Share your use cases and results

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Built on **LLaMA-3.1** (Meta AI)
- Uses **RoBERTa** (Facebook AI)
- Inspired by **Perceiver** architecture
- QLoRA implementation from **PEFT** library

---



**Star ⭐ this repo if you find it useful!**

