---
type: resource
status: active
created: 2026-07-02
updated: 2026-07-02
tags: [methodology, controls, experimental-design, causal-inference]
related:
  - [[endpoint-predictability]]
  - [[02-research-papers-llm-effect]]
  - [[decision-confound-defense]]
needs-review: false
---

# Methodology: Formal Specification of Controls

## Overview

This document specifies the formal implementation of controls used across the endpoint-predictability projects. Controls fall into three categories:

1. **Baseline controls** (isolate signal from noise)
2. **Null controls** (ensure signal is real, not artifact)
3. **Design controls** (separate mechanisms, identify causation)

---

## 1. Baseline Controls

### 1.1 BOS (Beginning-of-Sequence) Baseline

**Purpose**: Remove intrinsic text properties (rare words, unusual syntax) from novelty/resonance measures.

**Specification**:
```
For a passage T_t:

s̄(T_t | BOS) = mean surprisal when context is empty

Novelty(T_t) = s̄(T_t | context) − s̄(T_t | BOS)
             = contextual surprisal − intrinsic surprisal
```

**Implementation**:
```python
def compute_bos_baseline(text, model, tokenizer):
    """
    Compute surprisal of text with no context.
    """
    # Tokenize without prior context
    inputs = tokenizer(text, return_tensors='pt')
    
    # Forward pass (context is just <BOS> token)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss  # mean cross-entropy
    
    return loss.item()
```

**When to apply**:
- Novelty (always)
- Terminal resonance (recommended, to subtract out turn-intrinsic surprisal)
- Transience (optional; use to isolate contextual contribution)

**Why it matters**: Without it, rare-word passages look high-novelty just because they're unusual words, not because they're novel *given context*.

---

### 1.2 Correction for Context-Length Bias

**Purpose**: Shorter and longer passages don't accumulate surprisal fairly (longer passages lower surprisal just by having more words to condition on).

**Specification**:
```
For context C_t = all text up to time t:

Raw effect: ΔS_raw(t) = s̄(End | C_{t-1}) − s̄(End | C_t)

Length-corrected effect:
ΔS_corrected(t) = ΔS_raw(t) / n_tokens(T_t)

where n_tokens(T_t) is the token count of the segment added at t
```

**Interpretation**: ΔS_corrected is "surprisal-drop per token added," so longer passages don't automatically win.

**Alternative**: Use segment-level analysis (per-section) instead of per-token to avoid this issue entirely.

---

## 2. Null Controls

### 2.1 Permutation Control (Shuffle Test)

**Purpose**: Establish that signal is due to *narrative order*, not just bag-of-words content.

**Specification**:

For a novel or paper:

1. **Real ordering**: Compute curve S(t) and marginal contributions ΔS(t) in true order
2. **Shuffled ordering**: Randomly permute paragraphs/sections; recompute curve and ΔS
3. **Repeat**: Do K=1000 permutations

**Metrics to compare**:
```
Real vs. Shuffled:

Feature 1: Slope of cumulative curve [Σ ΔS(t)] vs. position
  Real should be steeper than mean(shuffled slope)

Feature 2: R² of linear fit to cumulative curve
  Real should have higher R² (more structured) than shuffled

Feature 3: Correlation between ΔS and position
  Real might show correlation (decreasing ΔS or other pattern);
  shuffled should be near zero

Feature 4: Max local ΔS
  Real might have one high-ΔS section (the reveal);
  shuffled should be more uniform
```

**Statistical test**:
```
For each feature f:
  f_real = computed on real ordering
  f_shuffled = list of computed features from K shuffles
  
  p-value = (n_times f_shuffled[i] >= f_real) / K
  
If p < 0.05, conclude: real ordering produces significantly 
different (usually more structured) curves than shuffled.
```

**Example code**:
```python
def permutation_test(sections, model, tokenizer, n_shuffles=1000):
    """
    Test whether real ordering matters.
    """
    # Real ordering
    features_real = compute_curve_features(sections, model, tokenizer)
    
    # Shuffled orderings
    features_shuffled = []
    for _ in range(n_shuffles):
        shuffled = random.sample(sections, len(sections))
        f = compute_curve_features(shuffled, model, tokenizer)
        features_shuffled.append(f)
    
    # Compare (e.g., R² of linear fit)
    p_value = (np.array(features_shuffled) >= features_real).mean()
    return p_value
```

**Interpretation**:
- p < 0.05: Order matters; signal is not pure content
- p > 0.05: No evidence that order contributes structure (concerning for narrative claims)

---

### 2.2 Decoy-Ending Control

**Purpose**: Ensure that the true ending becomes progressively more predictable (not all endings equally predictable).

**Specification**:

For a novel with true ending End_true:

1. **Alternative endings**: Create K=3–5 decoy endings:
   - Ending from a different novel (similar genre)
   - LLM-generated alternative (prompt: "write an alternative ending")
   - Neutral text (e.g., encyclopedia entry on related topic)

2. **Compute divergence**:
```
S_true(t) = mean surprisal of true ending given context C_t
S_decoy_i(t) = mean surprisal of decoy ending i given context C_t

Gap(t) = S_decoy(t) − S_true(t)
```

3. **Expected pattern**:
```
Early in the novel: Gap ≈ small (true and decoy endings similar)
Late in the novel: Gap ≈ large (true ending much more predictable)
Cumulative gap should increase monotonically
```

**Statistical test**:
```
Slope of Gap(t) vs. position should be significantly positive.
Correlation(position, Gap) should be > 0.5 and p < 0.05.

Shuffle control: If positions are shuffled, this correlation should vanish.
```

**Why this matters**: Without it, low surprisal of the ending could reflect that the ending is inherently predictable (or generic), not that the text set it up.

---

### 2.3 Genre Baseline (Optional)

**Purpose**: Establish typical curve shapes per genre, to validate typologies.

**Specification**:

For each genre (mystery, literary fiction, thriller, etc.):

1. Compute surprisal curves for N=5–10 novels in that genre
2. Plot mean and std curves
3. Compare to:
   - Random baseline (shuffled novels)
   - Cross-genre mixing (take chapters from different genres; recompute)

**Expected patterns**:
- Mystery: Sharp late-drop (reveal resolves surprise)
- Literary: Diffuse, gradual resolution
- Thriller: Multiple inflection points (plot twists)
- Science fiction: Moderate early drop (world is explained), then stays flat

---

## 3. Design Controls

### 3.1 Within-Author Panel (Causal)

**Purpose**: Isolate LLM-effect from confounds (field evolution, venue change, author population shift).

**Specification**:

1. **Author cohort**: Identify researchers with:
   - ≥1 paper pre-2022 (published or arXiv before 2022-12-31)
   - ≥1 paper post-2022 (published or arXiv 2023-01-01+)
   - Same field (minimize domain shift)

2. **Matching criteria**:
   - Same author (controls for style, expertise)
   - Ideally same venue (controls for acceptance standards)
   - If venue differs, control statistically

3. **Difference computation**:
```
For author i:

Feature_pre = mean curve features (pre-2022 papers)
Feature_post = mean curve features (post-2022 papers)

ΔFeature_i = Feature_post − Feature_pre
```

4. **Statistical test**:
```
H0: mean(ΔFeature) = 0 across all authors
H1: mean(ΔFeature) ≠ 0

Use paired t-test (per-author pairs) or mixed-effect model:

  Feature ~ post_2022 + (1 | author_id) + (1 | venue_id)
  
where post_2022 is the fixed effect of interest.
```

**Mixed-effect specification**:
```R
lmer(mean_ΔS ~ post_2022 + (1 | author) + (1 | venue), 
     data = paper_features)

# Report: Fixed effect (post_2022 coefficient) ± SE, p-value
# Random effects: Between-author and between-venue variance
```

**Causal interpretation**:
- If effect persists within-author, it's not due to author-level factors
- It could still be due to field-level changes, but less likely
- Removes the most obvious confounds

**Required sample size**:
- Goal: 50–100 author pairs
- Rule of thumb: 0.8 power to detect d=0.3 (medium effect): ~n=50 pairs

---

### 3.2 Placebo Trend (Temporal)

**Purpose**: Prove that post-2022 shift is a discontinuous break, not continuation of pre-existing drift.

**Specification**:

1. **Bin papers by year**: 2010, 2011, ..., 2021, 2022, 2023, 2024
2. **Compute curve features per year**:
```
mean_ΔS_by_year = [mean_ΔS(2010), mean_ΔS(2011), ..., mean_ΔS(2024)]
variance_by_year = [variance(2010), ..., variance(2024)]
```

3. **Fit piecewise-linear model** with potential knot at 2022:
```
Feature(year) = {
  β₀ + β₁ × year,                if year ≤ 2022
  β₀ + β₁ × year + α × (year − 2022),  if year > 2022
}

where α captures the post-2022 "shift" relative to pre-2022 trend
```

4. **Test for knot**:
```
H0: α = 0 (no discontinuity; straight line 2010–2024)
H1: α ≠ 0 (discontinuity at 2022)

Use likelihood-ratio test (compare nested models with/without knot)
or Chow test for structural break.
```

**Example (Python)**:
```python
from scipy.optimize import curve_fit

def piecewise_linear(year, beta0, beta1, alpha):
    """Piecewise-linear model with knot at 2022."""
    x = np.array(year)
    return np.where(x <= 2022, 
                    beta0 + beta1 * x,
                    beta0 + beta1 * x + alpha * (x - 2022))

# Fit
params, _ = curve_fit(piecewise_linear, years, features)
beta0, beta1, alpha = params

# Test significance of alpha
# (would need residuals, standard errors, etc.)
```

**Interpretation**:
- **Significant α, α > 0**: Pre-2022 trend continues, then **accelerates** post-2022 (LLM amplifies existing drift)
- **Significant α, α < 0**: Pre-2022 trend reverses post-2022 (LLM opposes existing pattern; unusual)
- **Insignificant α**: No discontinuity; papers have been drifting smoothly 2010–2024 (no evidence of LLM-specific effect)

---

### 3.3 Section-Normalized Comparison (Structural)

**Purpose**: Account for changes in paper length and IMRaD structure over time.

**Specification**:

Rather than comparing raw surprisal across papers, normalize by section position:

```
For section i in a paper (e.g., i=1 for Intro, i=2 for Methods, ...):

  ΔS_normalized(i) = ΔS(i) / total_n_sections

  Curve: [ΔS_norm(1), ΔS_norm(2), ..., ΔS_norm(n)]
```

**Benefit**: A paper with 5 sections and one with 8 sections are now directly comparable (normalized by their own structure).

**Mixed-effect model**:
```R
lmer(ΔS ~ post_2022 + section_position + (1 | paper_id), 
     data = section_surprisals)

# This captures the effect of LLM era while controlling for 
# the natural variation across sections
```

**Expected interpretation**:
- If post_2022 coefficient is significantly negative: papers in the LLM era have steeper resolution (more negative ΔS)
- Robust to changes in paper length or section count

---

## 4. Model-Validation Controls (For Surprisal Measurement)

### 4.1 Scorer-Ranking Stability

**Purpose**: Confirm that relative surprisal rankings are stable across scorers (robustness of rankings despite different absolute values).

**Specification**:

For two scorers (e.g., GPT-2 and Claude):

```
For each paper i, compute:
  ΔS_gpt2(i) = mean surprisal change toward conclusion (GPT-2)
  ΔS_claude(i) = mean surprisal change toward conclusion (Claude)

Compute Spearman rank correlation:
  ρ = rank_correlation(order_by_gpt2, order_by_claude)

Expectation: ρ > 0.7 (rankings are highly correlated)
```

**Implementation**:
```python
from scipy.stats import spearmanr

delta_s_gpt2 = [...]  # one value per paper
delta_s_claude = [...]

rho, pvalue = spearmanr(delta_s_gpt2, delta_s_claude)
print(f"Spearman ρ = {rho:.3f}, p = {pvalue:.4f}")
```

**Interpretation**:
- ρ > 0.7: Scorers agree on rankings; relative patterns are robust
- ρ < 0.5: Scorer-dependence is severe; claims are not robust
- 0.5 ≤ ρ ≤ 0.7: Moderate agreement; recommend using both scorers for robustness

---

### 4.2 Bits-Per-Byte Validation

**Purpose**: Ensure tokenizer-independent metric (bits-per-byte) is stable and comparable.

**Specification**:

For a text sample:

```
Compute surprisal three ways:
  1. Bits-per-token (GPT-2 tokenizer)
  2. Bits-per-byte (tokenizer-independent)
  3. Bits-per-token (Claude tokenizer)

Metric 2 should be similar across both models;
Metrics 1 and 3 should differ but be monotonically related.
```

**Check**:
```python
def validate_bpb(text, model_gpt2, tokenizer_gpt2, 
                 model_claude, tokenizer_claude):
    """Confirm BPB stability across tokenizers."""
    
    # GPT-2: bits-per-token
    s_gpt2 = compute_surprisal(text, model_gpt2, tokenizer_gpt2)  # bits/token
    
    # Convert to bits-per-byte
    n_tokens_gpt2 = len(tokenizer_gpt2.tokenize(text))
    n_bytes = len(text.encode('utf-8'))
    bpb_gpt2 = s_gpt2 * n_tokens_gpt2 / n_bytes
    
    # Claude: bits-per-token
    s_claude = compute_surprisal(text, model_claude, tokenizer_claude)
    
    # Convert to bits-per-byte
    n_tokens_claude = len(tokenizer_claude.tokenize(text))
    bpb_claude = s_claude * n_tokens_claude / n_bytes
    
    # Check agreement
    print(f"GPT-2 BPB:  {bpb_gpt2:.3f}")
    print(f"Claude BPB: {bpb_claude:.3f}")
    print(f"Difference: {abs(bpb_gpt2 - bpb_claude):.3f}")
    
    # Expect: difference < 0.1 BPB (reasonable agreement)
```

---

## 5. Summary Checklist

### Before Starting Analysis

- [ ] **BOS baseline** implemented and tested on sample texts
- [ ] **Permutation null** code written and validated (compare real vs. shuffled on toy example)
- [ ] **Decoy-ending control** designed (chosen decoy sources)
- [ ] **Within-author cohort** identified (50–100 authors with pre/post papers)
- [ ] **Placebo trend** specification confirmed (year bins, piecewise-linear model)
- [ ] **Scorer-ranking stability** test implemented (Spearman ρ computation)
- [ ] **Bits-per-byte** validation done on sample (GPT-2 vs. Claude agreement)

### During Analysis

- [ ] Compute each control in parallel with main analysis
- [ ] Report null-test results prominently (not in appendix)
- [ ] Flag if any control fails (e.g., permutation doesn't show difference)
- [ ] Interpret within-author effect separately (causal) from cross-sectional (correlational)

### In writeup

- [ ] Describe each control's purpose and specification
- [ ] Report effect size AND control result (e.g., "mean ΔS increases 0.3 bits post-2022 (p=0.02); permutation null rejected with p<0.001")
- [ ] Note limitations (e.g., "within-author sample n=50, moderate power")

---

## Related Resources

- [[novelty-transience-resonance-framework]]: Formal definitions of quantities being measured
- [[decision-confound-defense]]: How controls fit into the confound strategy
- [[02-research-papers-llm-effect]]: Main project where most controls are used

---

**Last updated**: 2026-07-02
**Maintainer**: Halfdan
**Status**: Specification complete; ready for implementation
