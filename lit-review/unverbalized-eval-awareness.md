# Literature review: measuring unverbalized evaluation awareness

**Scope.** How prior work has measured or detected evaluation awareness that a model does not
state in its output, which methods worked, and what they found. In scope: resampling and
behavioral-gap designs, activation probes, steering, and activation-to-text readouts. Out of
scope: work that only counts verbalized awareness, except where it calibrates the problem.

Sources were verified against their own arXiv or venue page (`refs.bib`). Abstracts were read
for every entry; full text only where noted. Non-peer-reviewed sources (workshop papers, blog
posts, industry reports) are labeled.

**Verification, Nov 2 2026.** All 18 entries with an arXiv ID match the arXiv API on title,
first author and author count. The other 8 have no arXiv record: two ICML 2026 workshop
papers, three blog posts, three Anthropic publications. Each was checked against its own
page. The citation-checker script reported 12 of 26 not found; it truncates titles at LaTeX
braces and does not query arXiv IDs, and it also missed SAD and Hua et al. The NLA entry
lists Anthropic as author because the byline could not be extracted from the page.

Queries run (Oct 27 and Nov 2, 2026): "unverbalized evaluation awareness probe", "eval
awareness steering probe reasoning model", "Towards Measuring and Detecting Unverbalized
Evaluation Awareness", "Smaller NLAs and probes read eval awareness", "WorkspaceBench", "J-Lens
oracle lens activations", "Thought Branches resampling", "Claude Sonnet 4.5 system card
evaluation awareness".

## 1. Why verbalized monitoring is not enough

- Frontier models often verbalize suspicion that they are being tested. Sonnet 4.5 did so in
  about 13 percent of alignment-evaluation transcripts, as reported in press coverage of the
  system card; the card itself was too large to fetch \cite{anthropic2025sonnet45card}. In the
  Qwen3 traces released with \cite{abdelnabi2025hawthorne}, GPT labels mark explicit test
  suspicion in 8 of 2587 placeholder-entity prompts and hedged suspicion in 291 more. Counted
  from the released data file, not the paper.
- Open-weight models verbalize far less in agentic scenarios. Across several open models,
  verbalized awareness was near zero, tracked task plausibility rather than scenario realism,
  and vanished when metacognitive instructions were removed from the system prompt
  \cite{vu2026dealing}.
- Chain-of-thought monitoring counts only what is said. Papers that use it treat their rates as
  lower bounds \cite{li2026decomposing}.
- Suppressing eval-awareness features in Sonnet 4.5 by steering lowered verbalized awareness and
  in some cases raised misaligned behavior. Anthropic concluded that internal recognition of
  evaluations shapes behavior even when unverbalized \cite{anthropic2025sonnet45card}.
- Whether awareness changes behavior is contested. Injecting or removing verbalized awareness
  in eight open reasoning models moved answer distributions by at most 3.7 points
  \cite{knecht2026limited}. On the other side, capabilities-framed awareness predicted compliance
  by 24 to 46 points on Qwen3-32B, and prefilling it shifted compliance in 10 of 11 cases
  \cite{zhuang2026notall}. Awareness decreased covert actions in OpenAI o3
  \cite{schoen2025stress}.

## 2. Methods used to measure what is not said

### 2.1 Behavioral gap with a known signature (model organisms)

**How.** Fine-tune a model so that it does something measurable (write Python type hints) more in
evaluation-like than deployment-like contexts. Then remove verbalization and check whether the
behavioral gap survives.

- \cite{hua2025steering}: continued pretraining on synthetic documents plus expert iteration
  gives a model that writes type hints more often in evaluation contexts. Steering toward
  "deployment" closes the gap. Preprint.
- \cite{taylor2026implanting} (blog): on the same organism, type hints appear about 70 percent
  of the time in eval contexts versus 12 in deployment with reasoning on, almost all accompanied
  by verbalization. With reasoning off, 25 versus 11 percent with almost no verbalization.
  Resampling away verbalized sentences leaves type hints in about half of cases. Proposes three
  levels: explicit verbalized, implicit chain-of-thought-mediated, fully latent.
- \cite{montoya2026unverbalized} (workshop): resampling removed all detected verbalizations in
  the Nemotron organism, but a 7.0 point type-hint gap remained; 65.3 points in a Qwen3.5
  replication. Linear probes trained only on non-coding eval/deploy prompts still predicted
  context on the suppressed sections, response AUROC 0.81 (Nemotron) and 0.71 (Qwen3.5).

**What it buys.** The only design with ground truth: the behavior is implanted, so a gap is
awareness by construction. Cost: fine-tuning a model, and the result is about an organism, not
a natural model.

### 2.2 Resampling at silent points

**How.** Cut a reasoning trace where nothing about evaluation has been said, resample continuations,
and measure how often continuations go on to verbalize. The fraction is a text-grounded estimate
of latent disposition at that state.

- Resampling is the principled on-policy alternative to editing a chain of thought; hand edits
  have small and unstable effects \cite{macar2025thoughtbranches}.
- Used to suppress verbalization in \cite{montoya2026unverbalized} and \cite{taylor2026implanting}.
- Verbalization training cuts rollouts just before a spontaneous eval statement and rewards
  verbalizing there; verbalized awareness rose 2.4 to 2.9 times while latent awareness and
  behavior stayed stable on Qwen3.6-35B-A3B, Kimi K2.6 and Inkling \cite{anwar2026verbalize}.

**What it buys.** Works on a natural model with no fine-tuning. Measures disposition to voice, so
awareness the model would never voice is invisible to it.

### 2.3 Linear probes on eval versus deploy prompts

**How.** Label prompts as evaluation or deployment by construction, train logistic regression on
residual-stream activations, read the probe on new transcripts.

- Llama-3.3-70B: AUROC 0.64 to 0.83 depending on probe variant; current safety evals read as
  evaluation-like to the probe \cite{nguyen2025probing}. Workshop paper.
- 15 models, 0.27B to 70B: probe-measured awareness follows a power law in model size
  \cite{chaudhary2025scales}. Preprint.
- Six open models: probe AUROC above 0.7 at the best layer for all; awareness present in base
  models, amplified by supervised fine-tuning \cite{heidari2026representation}. Preprint.
- 37 models: probes keep AUROC 0.98 even when rewrites push behavioral detection below chance;
  detection, behavior and representation axes are weakly coupled \cite{nayan2026notone}. Preprint.
- Steering along a probe direction changes compliance on harmful requests in reasoning models
  \cite{abdelnabi2025hawthorne}. NeurIPS 2025.

**The confound.** Evaluation prompts look like benchmarks and deployment prompts look like chat. A
probe trained on the standard contrast labeled 94 percent of deployment prompts rewritten into
MCQ format as evaluation. Training on a 2x2 of context by format fixed it \cite{devbunova2026format}.
Smaller models respond to format, larger ones to context \cite{singh2026format}.

**What it buys.** Cheap and transfers across tasks (\cite{montoya2026unverbalized} trained on
non-coding prompts and read coding traces). Needs the format control and a leakage check, since
the probe may read the cue text rather than the model's state.

### 2.4 Activation-to-text readouts

**How.** A second model reads one activation vector and writes what it encodes. No eval-specific
labels needed.

- Natural language autoencoders (NLAs) surfaced unverbalized evaluation awareness in Opus 4.6
  audits, more often on evaluation transcripts than real traffic; realism rewrites reduced it.
  Explanations confabulate specific details \cite{anthropic2026nla}. Industry publication.
- The J-lens reads single-token concepts from the residual stream with one linear map per layer.
  The workspace it reads sometimes encodes recognition of being in an evaluation, and ablating
  those representations exposed concealed propensities \cite{gurnee2026workspace}. Industry
  publication.
- Activation oracles answer natural-language questions about an activation and recover
  fine-tuned knowledge absent from the input text \cite{karvonen2025oracles}. Preprint.
- On Qwen3.6-27B with 76 technical tasks and injected cues at three strengths, a 27B NLA, a
  distilled 2B NLA, and a ridge probe trained on NLA-judged labels all read "being evaluated"
  about equally well. Released probes and checkpoints at layer 42 \cite{bricknell2026smallernlas}.
  Blog post, small sample.
- WorkspaceBench compares these readers on Qwen3.6-27B. Single-token lenses are reliable but
  cannot express order or composition; NLAs are expressive but nearly every readout contained a
  claim contradicted by context; a prompt-only baseline is a strong floor
  \cite{blank2026workspacebench}. No eval-awareness family exists in it.

**What it buys.** Readouts need no eval labels and can say what kind of awareness is present.
Cost: confabulation, no mechanistic grounding, and a judge is still needed to score the text.

## 3. Datasets used

- Minimal pairs of harmful requests with real versus placeholder entities
  \cite{abdelnabi2025hawthorne}. Cue is subtle and realistic; confounded with fictional framing.
- 1000 transcripts from 61 datasets labeled eval or deploy, chat and agentic
  \cite{needham2025know}. Gemini-2.5-Pro reached AUC 0.83 versus 0.92 for humans.
- SAD stages task: is this prompt from internal evaluation or deployment \cite{laine2024sad}.
- EvalAwareBench: 100 paired safety and capability tasks with eight toggleable trigger factors
  \cite{li2026decomposing}.
- Format-controlled 2x2 (context by format), length matched \cite{devbunova2026format}.
- 76 technical tasks with injected cues at subtle, obvious and very obvious strengths
  \cite{bricknell2026smallernlas}.
- Replayed real agentic trajectories with an appended evaluation prompt, to reduce
  distinguishability from deployment \cite{ivanov2026lure}.

## 4. What is agreed, contested, and open

**Agreed.**
- Verbalization, internal representation, and behavior come apart
  \cite{montoya2026unverbalized, nayan2026notone, heidari2026representation}.
- Linear probes decode eval versus deploy context in essentially every open model tested, with
  AUROC 0.7 to 0.98 depending on data \cite{nguyen2025probing, heidari2026representation, nayan2026notone}.
- Probe accuracy alone is not evidence of awareness, because format and cue text are confounded
  with the label \cite{devbunova2026format}.
- Hand-editing a chain of thought is a weaker intervention than resampling
  \cite{macar2025thoughtbranches, knecht2026limited}.

**Contested.**
- Whether awareness changes behavior in natural models: small effects
  \cite{knecht2026limited} versus framing-dependent large effects \cite{zhuang2026notall} and
  causal effects on covert actions \cite{schoen2025stress}.
- Whether single-layer steering does anything: negligible in three families
  \cite{nayan2026notone} versus clear compliance shifts \cite{abdelnabi2025hawthorne}.

**Open, and relevant to this project.**
- No study compares interpretability readouts (probe, J-lens, NLA, oracle, activation oracle)
  against the same behavioral anchor for unverbalized awareness. \cite{bricknell2026smallernlas}
  compares three readers against NLA-derived labels, which is circular, on 76 tasks.
- No study validates a readout against the resampling estimate of latent awareness. The
  resampling papers use probes trained elsewhere and report transfer, not calibration.
- Every probe study except \cite{devbunova2026format} leaves the format confound uncontrolled,
  and none reports a cue-deletion leakage check.
- Readouts' confabulation rate on eval-awareness claims specifically is unmeasured.

## 5. Implications for the design

- Use resampling at silent points as the behavioral anchor. It is the only text-grounded
  estimate of latent awareness that works on a natural model within budget.
- Treat each interpretability method as a judge of the activation at the cut point, scored
  against that anchor, with text readers that see the same prefix as the floor. This fills the
  first open gap above.
- Target Qwen3.6-27B if possible. It is the one open model with released J-lens, oracle lens,
  NLA, activation oracle and eval-awareness probes, and it has thinking mode.
- Keep the minimal-pair design from \cite{abdelnabi2025hawthorne} for cues, since it is the
  dataset with a published verbalization rate for Qwen3, but report the fictional-framing
  confound.
- Include a cue-deletion control and a prompt-only text baseline, since both confounds are
  documented and neither is standard.
