# REFRAG Quick Start Guide

## 🚀 TL;DR

REFRAG compresses massive documents into tiny KV prefixes, enabling LLMs to "remember" 100K+ tokens while using only a fraction of the memory. Think of it as **RAM vs. Hard Drive** for LLMs—fast compressed memory with selective expansion when needed.

---

## 📋 What We Have

### Two Implementations

| File | Method | VRAM | Speed | Best For |
|------|--------|------|-------|----------|
| `refrag_qLORA.ipynb` | 4-bit + LoRA | ~8 GB | Fast | Development, Prototyping |
| `refrag_full_precision.ipynb` | BF16 Full | ~60 GB | Slow | Production, Max Quality |

Both follow the same pipeline: **Reconstruction → CPT → SFT → Inference**

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -q torch transformers accelerate datasets peft bitsandbytes faiss-cpu tqdm
```

### 2. Prepare Your Data

#### CPT Corpus (`corpus.jsonl`)
```jsonl
{"id": "1", "text": "Your training documents here..."}
{"id": "2", "text": "More documents..."}
```

#### SFT Dataset (`sft.jsonl`)
```jsonl
{"inputs": "Question or instruction", "targets": "Expected response"}
```

#### Inference Passages (`passages.jsonl`)
```jsonl
{"id": "1", "text": "Knowledge base documents..."}
```

### 3. Configure & Run

```python
# Edit config in notebook
cfg = {
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "corpus_path": "path/to/corpus.jsonl",
    "sft_path": "path/to/sft.jsonl",
    "passages_path": "path/to/passages.jsonl",
    "epochs_recon": 1,
    "epochs_cpt": 1,
    "epochs_sft": 1,
}

# Run all stages
run_reconstruction(cfg)  # Learn to compress
run_cpt(cfg)             # Learn to use compressed context
run_sft(cfg)             # Task-specific fine-tuning
run_infer(cfg)           # Answer questions with compressed memory
```

---

## 🧠 How It Works (Simple Explanation)

### Before REFRAG
```
Question: "What causes BGP flapping?"
↓
Retrieve: 100 relevant documents (50K tokens)
↓
LLM: "Too many tokens! I can only handle 8K!"
↓
Result: Truncate or summarize (lose information)
```

### With REFRAG
```
Question: "What causes BGP flapping?"
↓
Compress: 100 documents → 100 tiny KV prefixes (1K "virtual tokens")
↓
LLM: "I can handle this easily!"
↓
Selective Expand: "I need raw text for chunks 5, 23, 67" (high attention)
↓
Result: Accurate answer using full knowledge base
```

---

## 🎯 Key Concepts

### 1. **Compression**
```python
Original: "BGP uses TCP port 179 and is defined in RFC 4271..."  # 16 tokens
Compressed: [KV prefix]  # 1 "virtual token"

# 16:1 compression ratio!
```

### 2. **Projector**
```python
# The magic component that learns compression
Text → RoBERTa → Embedding (768-dim) → Projector → KV Prefixes
                                         ↑
                              This learns how to compress!
```

### 3. **Selective Expansion**
```python
# Model monitors its own attention
if attention[chunk_42] > threshold:
    inject_raw_text(chunk_42)  # "I need more details on this!"

# Smart, dynamic retrieval instead of blind concatenation
```

---

## 🔑 Configuration Cheat Sheet

### Critical Parameters

```python
cfg = {
    # Compression
    "prefix_len": 1,        # Tokens per compressed chunk (1-4)
    "chunk_size": 16,       # Tokens per chunk before compression (8-64)

    # Training
    "epochs_recon": 1,      # Reconstruction epochs (1-3)
    "epochs_cpt": 1,        # CPT epochs (1-3)
    "epochs_sft": 1,        # SFT epochs (1-3)
    "lr_recon": 2e-4,       # Reconstruction learning rate
    "lr_cpt": 1e-4,         # CPT learning rate
    "lr_sft": 5e-5,         # SFT learning rate

    # Inference
    "expand_budget": 256,   # Max tokens for expansion (128-512)
    "max_new_tokens": 256,  # Max response length
}
```

### Tuning Guide

| Goal | Adjust |
|------|--------|
| Better compression | ↑ `prefix_len` (1→2), ↑ `epochs_recon` |
| Better accuracy | ↑ `expand_budget` (256→512), ↓ `chunk_size` (16→8) |
| Faster training | Use QLoRA version, ↓ batch size |
| More capacity | ↑ projector width in code (`width_mult=4→8`) |

---

## 📊 Expected Results

### Compression Performance
```
Input: 10,000 chunks × 16 tokens = 160,000 tokens
Output: 10,000 prefixes × 1 virtual token = 10,000 tokens
Compression Ratio: 16:1

Effective Context: 10,000 (compressed) + 256 (expanded) ≈ 10K
Actual Content: ~160K tokens of information!
```

### Training Time (QLoRA on A100)
```
Reconstruction: ~1 hour (1 epoch, 10K samples)
CPT: ~2 hours (1 epoch, 10K samples)
SFT: ~1 hour (1 epoch, 2K samples)
Total: ~4 hours for full pipeline
```

### Inference Speed
```
Without expansion: ~20 tokens/sec (mostly compressed)
With expansion: ~15 tokens/sec (selective expansion)
Baseline RAG: ~5 tokens/sec (full raw text)
Speedup: 3-4x vs. traditional RAG
```

---

## 🎓 Training Stages Explained

### Stage 1: Reconstruction (Warm-up)
**Goal**: Teach projector to compress without losing info

```python
run_reconstruction(cfg)
```

**What happens:**
- Encoder converts text to embeddings
- Projector learns to create KV prefixes
- LLM (frozen) tries to reconstruct text from prefixes
- Projector gets better at preserving information

**Output**: `projector.pt` (can compress text now!)

---

### Stage 2: CPT (Continual Pre-Training)
**Goal**: Teach LLM to use compressed context

```python
run_cpt(cfg)
```

**What happens:**
- Randomly mask 50% of labels (compression mixing)
- LLM must predict masked tokens using KV prefixes
- Both projector + LLM learn together
- LLM learns to "decompress" information from prefixes

**Output**: `lm/` (LoRA adapters) + `projector.pt` (improved)

---

### Stage 3: SFT (Supervised Fine-Tuning)
**Goal**: Task-specific instruction tuning

```python
run_sft(cfg)
```

**What happens:**
- Standard instruction fine-tuning
- No compressed context (pure task learning)
- Learns answer formatting, task patterns
- Combines with compression skills from CPT

**Output**: `sft_model/` (task-ready model)

---

### Stage 4: Inference
**Goal**: Answer questions using compressed memory

```python
run_infer(cfg)
```

**What happens:**
- Compress entire knowledge base
- Generate response token-by-token
- Monitor attention over compressed prefixes
- Selectively expand high-attention chunks
- Continue generation with expanded context

**Output**: Answer to your query!

---

## 🛠️ Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
```python
# Solutions:
1. Use QLoRA version instead of full precision
2. Reduce batch size: bs=8 → bs=4
3. Reduce max_len: 4096 → 2048
4. Use smaller model: 8B → 3B
```

### Issue 2: Poor Compression Quality
```python
# Solutions:
1. Train longer: epochs_recon=1 → epochs_recon=3
2. Increase prefix_len: 1 → 2
3. Increase projector capacity: width_mult=4 → width_mult=8
4. Check data quality (is text clean?)
```

### Issue 3: Slow Inference
```python
# Solutions:
1. Reduce expand_budget: 256 → 128
2. Increase chunk_size: 16 → 32 (fewer chunks)
3. Disable expansion: expand_budget=0
4. Use smaller encoder: roberta-base → distilroberta-base
```

### Issue 4: Low Answer Quality
```python
# Solutions:
1. Increase expand_budget: 256 → 512
2. Train longer CPT: epochs_cpt=1 → epochs_cpt=3
3. Better SFT data (more examples, higher quality)
4. Decrease chunk_size: 16 → 8 (finer granularity)
```

---

## 🎯 Use Case Examples

### Example 1: Network Diagnostics
```python
cfg = {
    "corpus_path": "network_configs.jsonl",      # Router configs, docs
    "sft_path": "network_qa.jsonl",              # Q&A pairs on networking
    "passages_path": "live_configs.jsonl",       # Current configs to diagnose
    "query": "Why is BGP flapping on router-01?",
}

run_infer(cfg)
# Model uses compressed memory of ALL configs to diagnose!
```

### Example 2: Code Understanding
```python
cfg = {
    "corpus_path": "codebase.jsonl",             # All .py files
    "sft_path": "code_qa.jsonl",                 # Code Q&A
    "passages_path": "new_pr_code.jsonl",        # Code to review
    "query": "Explain how authentication works in this PR",
}

run_infer(cfg)
# Model has "read" entire codebase in compressed form!
```

### Example 3: Medical Literature
```python
cfg = {
    "corpus_path": "pubmed_abstracts.jsonl",     # 1M papers
    "sft_path": "medical_qa.jsonl",              # Clinical Q&A
    "passages_path": "patient_case.jsonl",       # Patient info
    "query": "What treatments are recommended for this case?",
}

run_infer(cfg)
# Model synthesizes info from massive literature base!
```

---

## 📈 Performance Tips

### For Maximum Quality
```python
cfg = {
    "prefix_len": 2,              # More capacity per chunk
    "chunk_size": 8,              # Finer-grained chunks
    "epochs_recon": 3,            # Better compression
    "epochs_cpt": 3,              # Better retrieval
    "expand_budget": 512,         # More expansion
}
```

### For Maximum Speed
```python
cfg = {
    "prefix_len": 1,              # Minimal overhead
    "chunk_size": 32,             # Fewer chunks
    "epochs_recon": 1,            # Fast training
    "epochs_cpt": 1,              # Fast training
    "expand_budget": 128,         # Less expansion
}
```

### For Balanced Performance
```python
cfg = {
    "prefix_len": 1,
    "chunk_size": 16,
    "epochs_recon": 1,
    "epochs_cpt": 1,
    "epochs_sft": 1,
    "expand_budget": 256,
}
# This is the default—good starting point!
```

---

## 🧪 Testing Your Implementation

### Sanity Check 1: Compression Works
```python
# After reconstruction
run_reconstruction(cfg)

# Test: Can we reconstruct text from compressed prefixes?
test_text = "BGP uses TCP port 179"
embedding = encoder(test_text)
K, V = projector(embedding)
reconstructed = model.generate(past_key_values=[(K,V)])

if similar(reconstructed, test_text):
    print("✅ Compression works!")
else:
    print("❌ Need more training")
```

### Sanity Check 2: Retrieval Works
```python
# After CPT
run_cpt(cfg)

# Test: Can model answer using only compressed context?
compressed = compress_all("knowledge_base.jsonl")
answer = model.infer(compressed, "What is X?")

if answer_is_correct:
    print("✅ Retrieval works!")
else:
    print("❌ Need more CPT training")
```

### Sanity Check 3: Expansion Works
```python
# During inference
attention_logs = []

def monitor_attention(step, attn):
    attention_logs.append((step, attn))

run_infer(cfg, attention_callback=monitor_attention)

# Check: Did model expand high-attention chunks?
for step, attn in attention_logs:
    if attn.max() > threshold:
        print(f"✅ Expanded at step {step}")
```

---

## 📚 Documentation Reference

- **[README.md](README.md)**: Full technical overview, architecture, use cases
- **[META_REFRAG.md](META_REFRAG.md)**: Deep dive into meta-learning aspects
- **[QUICK_START.md](QUICK_START.md)**: This file—quick reference guide

---

## 🎓 Learning Path

1. **Understand Basics** → Read README.md Overview section
2. **Run Example** → Follow Quick Start guide (this file)
3. **Understand Meta-Learning** → Read META_REFRAG.md
4. **Customize** → Adapt to your use case
5. **Optimize** → Tune hyperparameters
6. **Deploy** → Production-ready inference

---

## 🤝 Getting Help

- **Questions**: Open GitHub issues
- **Bugs**: Report with reproducible examples
- **Ideas**: Share in discussions

---

## ✅ Checklist for New Users

- [ ] Installed dependencies
- [ ] Prepared data in JSONL format
- [ ] Ran reconstruction stage
- [ ] Ran CPT stage
- [ ] Ran SFT stage
- [ ] Tested inference
- [ ] Tuned hyperparameters
- [ ] Achieved target performance

---

## 🎉 You're Ready!

Start with the QLoRA notebook for quick experimentation, then move to full precision for production. Happy compressing! 🚀

---

**Remember**: REFRAG learns HOW to compress and retrieve, not just WHAT to remember. That's what makes it powerful! 💪
