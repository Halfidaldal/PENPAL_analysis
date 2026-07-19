---
type: decision
status: active
created: 2026-07-02
updated: 2026-07-02
tags: [methodology, model-selection, baselines, measurement]
related:
  - [[endpoint-predictability]]
  - [[decision-confound-defense]]
  - [[03-model-contamination-study]]
needs-review: false
---

# Decision: Scorer Selection for Endpoint-Predictability Framework

## Context

Surprisal-based measures (novelty, transience, terminal resonance) require a language model to assign probabilities. Which model should we use?

**Key constraint**: Different models assign different surprisals, so the choice affects absolute values. But we care about relative patterns and comparisons across time/conditions, so scorer choice should be transparent and strategic.

**Specific challenge**: For the research-paper LLM-effect project, we need to defend against confound that post-2022 models find post-2022 text less surprising due to distributional proximity. This makes scorer choice load-bearing.

---

## Candidate Models

### Pre-2021 Era (Uncontaminated Baseline)

| Model | Size | Released | Tokenizer | Availability | Pros | Cons |
|-------|------|----------|-----------|--------------|------|------|
| GPT-2 | 1.5B | 2019-02 | BPE (50k) | HuggingFace | Small, fast, well-known baseline | Very small; may have indexing artifacts |
| GPT-Neo 2.7B | 2.7B | 2021-03 | GPT-2 tokenizer | EleutherAI | Slightly larger; same tokenizer consistency | Still relatively small |
| GPT-J 6B | 6B | 2021-06 | GPT-2 tokenizer | EleutherAI | Reasonable size; standardized | Some ambiguity on training-data cutoff (late 2021, boundary region) |

**Caution**: Even pre-2021 models may have mild source bias from earlier LLM overlap (e.g., GPT-2 training data includes some synthetic/generated text). Literature (2026 retrieval papers) shows GPT-2 is cleaner than 2024 but not perfectly uncontaminated. Use as a baseline, not as "uncontaminated ground truth."

### Post-2023 Era (Recent Model)

| Model | Size | Released | Tokenizer | Availability | Pros | Cons |
|-------|------|----------|-----------|--------------|------|------|
| Claude 3.5 (base) | ~50B+ | 2024 | Custom (proprietary) | Anthropic API | Strong, modern capabilities; likely exposed to LLM text | Proprietary; not reproducible |
| LLaMA-2 7B | 7B | 2023-07 | SentencePiece | Meta | Open, trained with known cutoff (2023), larger than GPT-Neo | Smaller relative to modern standards |
| Mistral 7B | 7B | 2023-09 | SentencePiece | Open | Efficient; good instruction following | Relatively new; less tested |

**Rationale for choosing post-2023**: We want a model that represents "modern LLM" to show the full contrast with pre-2021. Claude 3.5 is the obvious choice if using Anthropic API; LLaMA-2 if reproducibility is critical.

### Transition Era (Optional, For Dose-Response)

| Model | Size | Released | Notes |
|-------|------|----------|-------|
| GPT-J 4B | 4B | 2021-06 | Boundary (late 2021) |
| OPT 6.7B | 6.7B | 2022-05 | Trained on OWT+BookCorpus+Code, explicit LLM-generation training |
| Pythia 6.9B | 6.9B | 2023-04 | Dedup versions track dataset size; good for dose-response |

**Optional**: Use Pythia dedup variants to measure how contamination scales with dataset size.

---

## Selection Decision

### Primary Design

**Scorer 1 (Pre-2021 baseline)**: GPT-2 1.5B
- Small, fast, widely available
- Tokenizer stable (no drift)
- Convenient for prototyping
- Accept the limitation: mild source bias is possible

**Scorer 2 (Post-2023 modern)**: Claude 3.5 (or LLaMA-2 7B as fallback)
- Claude 3.5: Strong capabilities, represents state-of-the-art
- LLaMA-2: If reproducibility/offline is critical

**Why this pair**:
- Maximum contrast (6 years, ~50x size difference)
- Tests the core hypothesis: does effect scale with scorer era?
- Practical (both accessible)

### Secondary (Optional, For Robustness)

**Scorer 3 (Transition era)**: Pythia 6.9B (dedup version)
- Allows dose-response fit
- Can estimate how much contamination matters
- Optional but recommended if analysis is tight

---

## Measurement Considerations

### Tokenizer Drift

Different models use different tokenizers (BPE, SentencePiece, proprietary), so per-token surprisal is not directly comparable.

**Mitigation**:
1. **Primary metric**: Use bits-per-byte (not bits-per-token)
   ```
   surprisal_bpb = (sum of surprisals in nats) / (n_bytes)
   convert to bits: / ln(2)
   ```
   This normalizes across tokenizers.

2. **Secondary metric**: Report per-token but flag tokenizer differences:
   ```
   "GPT-2 tokenizes into N tokens; Claude into M tokens. 
    Absolute surprisal not directly comparable, but rankings should be robust."
   ```

3. **Validation**: Confirm that token-wise ranking of passages is stable across scorers (Spearman ρ).

### Context Window

Different models have different max context windows:
- GPT-2: ~1024 tokens (can be extended, but standard is 1024)
- Claude 3.5: ~200k tokens
- LLaMA-2: ~4096 tokens

For novel-endings project (full books), context window might be limiting. For research papers (few thousand tokens), all are fine.

**Mitigation**:
- For papers: Use full papers without concern (all models fit)
- For novels: Use sliding-window or hierarchical approach (split into chapters; compute surprisal within chapters; aggregate)
- **Document the approach**: "Due to context limits, we compute surprisal per chapter and aggregate to novel-level features."

---

## Implementation Specifications

### Setup

```python
# GPT-2 (pre-2021 baseline)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# Claude 3.5 (post-2023)
import anthropic
client = anthropic.Anthropic(api_key="...")

# LLaMA-2 (optional transition)
model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### Inference Pipeline

```python
def compute_surprisal_bpb(text, model, tokenizer, device='cuda'):
    """
    Compute bits-per-byte surprisal (tokenizer-independent).
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss  # cross-entropy in nats
    
    # Convert: loss is cross-entropy (nats/token averaged)
    # Multiply by n_tokens to get total nats
    n_tokens = inputs['input_ids'].shape[1]
    total_nats = loss.item() * n_tokens
    
    # Convert to bits-per-byte
    n_bytes = len(text.encode('utf-8'))
    bpb = total_nats / n_bytes / torch.log(torch.tensor(2.0))
    
    return bpb
```

### Batch Processing

For large corpora (500+ papers), use batch inference:

```python
def compute_surprisal_batch(papers, model, tokenizer, batch_size=8):
    """
    Batch inference for efficiency (GPU utilization).
    """
    results = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        # Tokenize, compute in parallel
        ...
    return results
```

---

## Validation Checklist

Before committing to full analysis:

- [ ] **Inference speed**: Time GPT-2 on 100 research papers. Budget ~10–20 minutes. If slower, adjust batch size or use smaller model.
- [ ] **Stability**: Compute same paper's surprisal twice; confirm reproducibility (differ by <0.1%).
- [ ] **Sanity checks**:
  - Does GPT-2 assign reasonable surprisals? (Expected: 3–6 bits/word for typical text)
  - Do systematic variations make sense? (Rare words higher surprisal, common words lower)
  - Do different scorers rank papers similarly? (Spearman ρ > 0.7)

- [ ] **Memory**: Check VRAM for full model inference. GPT-2 fits on most GPUs; Claude via API. LLaMA may need quantization on smaller GPUs.

- [ ] **Error handling**: What happens on very long papers (>10k tokens)? Truncate, or sliding window?

---

## Scoring Decision for Each Project

### Novel-Endings Project

**Scorer**: GPT-2 (or Claude 3.5 for long-context stability)

**Justification**: 
- Novels can be long; GPT-2's 1024 context is limiting (use sliding-window or chapter-level)
- Claude's 200k context is overkill but clean
- For initial work, GPT-2 is fine; switch to Claude if running into context limits

### Research-Paper LLM-Effect Project

**Scorer pair** (mandatory):
- Primary: GPT-2 (pre-2021 baseline, defends against confound)
- Secondary: Claude 3.5 or LLaMA-2 (post-2023 modern)

**Why two**: The whole defense rests on showing effect differs by scorer era. One scorer is insufficient.

### Model-Contamination Study

**Scorer panel**:
- Pre-2021: GPT-2, GPT-Neo
- Transition: OPT (2022), Pythia (2023)
- Post-2023: Claude, LLaMA-2

**Why panel**: Measure dose-response (how much contamination scales with scorer era).

---

## Cost & Timeline Implications

| Scorer | Setup | Inference Cost | Availability |
|--------|-------|-----------------|--------------|
| GPT-2 | Free download | ~10 min/100 papers (GPU) | HuggingFace |
| Claude 3.5 | API key (costs $) | ~5 min/100 papers (via API) | Anthropic |
| LLaMA-2 | Free download | ~30 min/100 papers (GPU) | Meta/HuggingFace |

**For research-paper project** (500 papers × 2 scorers):
- GPT-2 + Claude: ~1 hour + API costs (~$10–50 depending on token usage)
- GPT-2 + LLaMA: ~1.5 hours GPU time, free

**Recommendation**: Use GPT-2 + Claude for cleanliness; switch to GPT-2 + LLaMA if cost is prohibitive.

---

## Handling Tokenizer Differences

### Strategy 1: Bits-Per-Byte (Recommended)

```
surprisal_bpb = (sum of per-token surprisals) / (n_bytes in UTF-8)
```

Advantages:
- Tokenizer-independent
- Directly comparable across models
- Standard in some ML literature

Disadvantages:
- Less common in NLP papers (reviewers might question it)
- Slightly less intuitive than bits-per-token

### Strategy 2: Normalize Per-Model

```
surprisal_normalized = (surprisal_model − mean) / std

where mean, std computed on same corpus with that model
```

Advantages:
- Preserves per-token interpretation
- Easy to explain (z-score standardization)

Disadvantages:
- Loses absolute-magnitude comparison across models
- Only relative patterns matter

**Recommendation**: Use Strategy 1 (bits-per-byte) as primary. Report Strategy 2 as sensitivity check.

---

## Open Questions

1. **Is GPT-2 truly "pre-LLM-contamination"?** 
   - Caveat: Mild source bias documented in retrieval lit. But it's the best available pre-2021 baseline.
   - Mitigation: Report it as one leg of a tripod (scorer + within-author + placebo); not the sole defense.

2. **Should we use instruction-tuned or base models?**
   - Decision: Base models. RLHF and instruction-tuning introduce style biases orthogonal to the question.
   - Claude 3.5 is fine (it's a quality model); avoid ChatGPT-3.5 (instruction-tuned, style-optimized).

3. **How to handle very long papers (>20k tokens)?**
   - Option A: Truncate to first 10k
   - Option B: Sliding window (compute surprisal per section independently, aggregate)
   - Option C: Hierarchical (compute per-paragraph, then roll up)
   - Decision: Use Option B (sliding window by IMRaD section) — most defensible.

---

## Decision Summary

| Project | Scorer 1 | Scorer 2 | Notes |
|---------|----------|----------|-------|
| Novel-endings | GPT-2 | (Claude 3.5 optional for long-context) | Simpler study; one scorer sufficient |
| Research-paper LLM-effect | GPT-2 | Claude 3.5 | **Non-negotiable pair** for confound defense |
| Model-contamination | GPT-2, Neo | OPT, Pythia, Claude | Panel for dose-response |

---

## Implementation Roadmap

### Week 1: Setup
- [ ] Download GPT-2 and LLaMA-2; confirm local inference works
- [ ] Set up Claude API and test token costs
- [ ] Write tokenizer-independent surprisal functions (bits-per-byte)
- [ ] Test on 10-paper pilot (all scorers)

### Week 2: Validation
- [ ] Run inference on 50-paper sample (full pipeline)
- [ ] Check ranking stability across scorers (Spearman ρ)
- [ ] Profile speed and memory; optimize batching if needed
- [ ] Sanity-check surprisal distributions (histogram by paper type)

### Week 3+: Full Analysis
- [ ] Run on full corpus with primary scorer (GPT-2)
- [ ] Run on full corpus with secondary scorer (Claude)
- [ ] Compare effect sizes; finalize conclusions

---

## Review & Iteration

**Before finalizing corpus analysis, confirm**:
- Is the bits-per-byte metric defensible in your field? (Consult advisor/collaborators)
- Have any new pre-2021 baseline models been released that are better than GPT-2?
- Is the Claude API cost acceptable, or pivot to open LLaMA?

---

## Related Decisions

- [[decision-confound-defense]]: Why we need two scorers for the research-paper project
- [[03-model-contamination-study]]: Using a full scorer panel to measure contamination dose-response

---

**Decision made**: 2026-07-02
**Status**: Finalized (pending Week-1 pilot confirmation)
**Reviewed by**: Halfdan
