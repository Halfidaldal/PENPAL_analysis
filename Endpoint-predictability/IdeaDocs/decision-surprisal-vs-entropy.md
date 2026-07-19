---
type: decision
status: active
created: 2026-07-07
updated: 2026-07-07
tags: [methodology, information-theory, surprisal, entropy, measurement]
related:
  - "[[endpoint-predictability]]"
  - "[[surprisal-entropy-perplexity-guide]]"
  - "[[novelty-transience-resonance-framework]]"
  - "[[decision-scorer-selection]]"
needs-review: true
---

# Decision: Surprisal (Not Entropy) Is the Correct Measure for This Framework

This note was a broken/phantom link in [[endpoint-predictability]]'s "Key References & Decisions" section (referenced since the project note was written, but never actually filed). Filed now as part of a vault-wide reference audit. Content below synthesizes reasoning that already existed in [[surprisal-entropy-perplexity-guide]] rather than introducing new claims — flagging `needs-review: true` since this is a reconstruction of an implied decision, not a decision you explicitly dictated to me; confirm it reflects your actual reasoning before treating it as settled.

## Context

The endpoint-predictability framework (and PENPAL before it) needs a measure of "how much did this passage/turn matter." Information theory offers two natural candidates: **entropy** (uncertainty before the text arrives) and **surprisal** (how unexpected the text that actually arrived was). They are related but not interchangeable, and picking the wrong one would measure a different phenomenon than the one the framework claims to measure.

## Decision

Use **surprisal**, not entropy, as the base measure for novelty, transience, resonance, and endpoint-predictability's ΔS/terminal resonance — everywhere the framework asks "how much did *this specific passage* matter."

## Rationale

Per [[surprisal-entropy-perplexity-guide]]'s "When to Use Each" section:

- **Surprisal is anchored to what was actually written.** `s(w|c) = −log₂ p(w|c)` is a property of the specific text that occurred, not of the situation before it was written. This is exactly what the framework needs: novelty asks "how surprising was this *specific* turn," not "how open was the moment before the turn happened."
- **Entropy measures something different: the openness of a situation, independent of what was actually said.** It's the right tool for questions like "at what points in the story do many outcomes seem possible?" — a genuinely different question from "did this passage do something unexpected?"
- **Novelty and transience specifically require surprisal because both ask about realized choices, not optionality.** The framework wants to reward — and measure — an agent for *picking* something unexpected, not for merely facing an uncertain moment. An entropy-based measure would be blind to whether the agent actually exploited that uncertainty or played it safe; two turns following an identical high-entropy context could have wildly different surprisal depending on what was actually written, and that difference is the entire object of study.
- **Terminal resonance (ΔS) is a marginal-surprisal-change measure by construction** — `S(t) = s̄(End | C_t)`, i.e., the *realized* surprisal of the known ending given context so far. Entropy has no natural equivalent here, since the ending is fixed and known, not an open distribution of possible futures — there's nothing left to be "uncertain" about in the way entropy measures.

## Consequences

- All of the framework's core quantities (novelty, transience, local/terminal resonance, ΔS) stay defined in terms of realized surprisal, consistent with how they're already specified in [[novelty-transience-resonance-framework]] — this decision doesn't change any existing formula, it documents *why* that choice was made.
- Entropy remains a valid, different tool the project could reach for separately if a future analysis specifically wants to characterize situational openness rather than realized choices (e.g., "does this genre open up more possibility space in Act 1 than Act 3?") — per the guide's "Use Entropy When" section. That would be a different research question, not a replacement for the surprisal-based core measures.
- [[decision-scorer-selection]] and [[03-model-contamination-study]]'s confound concerns apply specifically to surprisal (since it depends on a scorer model's probability estimates for actual text) — this decision is a prerequisite for those, not independent of them.

## Related

- [[endpoint-predictability]]
- [[surprisal-entropy-perplexity-guide]]
- [[novelty-transience-resonance-framework]]
- [[decision-scorer-selection]]
