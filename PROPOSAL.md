# Proposal: Evaluation awareness, verbalized and unverbalized

Assignment 2 (due Nov 9). Status: proposal, nothing built yet. Section 2 is the list of
decisions that need an answer before any code is written.

## 0. Short version

- Do the split (verbalized vs unverbalized), not "just unverbalized". The assignment
  requires an LLM judge that labels response text and a probe that predicts the judge.
  Unverbalized awareness is by definition absent from the text, so a text judge cannot
  label it directly. The split design gives you a judge-validatable label and, as a
  by-product, the unverbalized subset. It is the same pipeline with two judge passes.
- The main scientific move: get a label for every transcript by appending a follow-up
  question after the response ("Was this interaction an evaluation?") and judging that
  answer. Then probe activations taken from the original response, before the question
  was asked. The subset where the follow-up says "eval" but the response never mentioned
  it is your unverbalized-awareness set. This is the closest you can get to the ICML 2026
  "unverbalized" papers while staying inside the assignment's judge-then-probe structure.
- Cut the method list down. The required core is: LLM judge, linear probe, majority
  baseline, text baseline. Add at most two of: SAE-feature probe, Activation Oracle
  readout, mean-difference direction. Drop natural language autoencoders unless the
  target model is one of the four with released NLAs. Grading weights validation and
  analysis, not method count.
- The biggest validity risk is the format confound (Devbunova 2026): eval prompts look
  like benchmarks and deploy prompts look like chat, so a probe learns format. Build the
  dataset as a 2x2 (context x format) from the start.
- Second biggest risk is leakage (the LessWrong SIEVE post): if you induce verbalization
  with an explicit "you are being tested" cue, the probe reads the cue from the prompt,
  not the model's state. Cue-deletion is the control.

## 1. What the assignment constrains

Pulled from the PDF; these shape every design choice below.

- Target model must be open-weight with accessible activations. Budget $150 to $200.
- Judge labels come from an LLM judge with a rubric, validated against a manually
  labeled sample. Binary or few-class.
- Probe predicts the judge's label from activations at the beginning and end of the
  generated response, at multiple layers. Preprocessing fit on train only.
- Required baselines: majority class, TF-IDF or bag-of-words logistic regression on the
  response text. "Any other baseline you think appropriate."
- One control for a plausible alternative explanation. Examples that defeat surface cues.
- Splits by underlying scenario. Test set untouched until judge and probe are frozen.
- One modest generalization check (held-out template or topic), can live inside the
  test split.
- Manual review of agreements and disagreements with labels assigned blind.
- Report must answer: what implicit geometric assumption does your probe make.
- "Use the provided infrastructure to extract an activation vector." We have not seen
  this infrastructure. It may pin the target model and the extraction format.

## 2. Decisions needed from you before building

Each has my recommendation. Reply with a number and a choice, or "other".

### D1. Scope: split, or unverbalized only
Recommendation: split. Reason in section 0. "Unverbalized only" is not buildable under
the judge-label requirement without a model organism with a known behavioral signature
(the Hua / Montoya type-hint setup), which needs fine-tuning and is out of budget.

### D2. Target model
Constraints: fits one GPU, activations accessible, and ideally has released
interpretability assets so the extra readouts cost nothing to train.

| Model | Thinking mode | Gemma Scope SAEs | Activation Oracle released | NLA released | GPU |
|---|---|---|---|---|---|
| Qwen3-8B | yes | no good ones known | yes | no | 24 GB |
| Gemma-2-9b-it | no | yes | yes | no | 24 GB |
| Llama-3.1-8B-Instruct | no | Llama Scope (base) | yes | no | 24 GB |
| Gemma-3-12b-it | no | Gemma Scope 2 (verify) | no | yes | 40 GB |
| Gemma-3-27b-it | no | Gemma Scope 2 (verify) | yes | yes | 80 GB |
| Qwen2.5-7B-Instruct | no | no | no | yes | 24 GB |

Recommendation: Qwen3-8B if you want the chain-of-thought layer (see D3); Gemma-2-9b-it
if you want SAE features and no CoT. Gemma-3-27b-it is the only one with all three
assets but needs an 80 GB card and roughly triples generation cost. Check whether the
course infrastructure pins this before deciding.

### D3. Chain of thought: on or off
If the target has thinking mode, verbalization has three levels: in the final answer, in
the CoT only, in neither. The "CoT only" level is where the 2025 to 2026 literature
found most of the action (Abdelnabi and Salem; the Knecht et al. null result on
behavior). It is the most interesting axis for unverbalized awareness. It costs: a
second judge pass over the CoT, a decision about whether "beginning of response" means
start of CoT or start of answer, and longer generations. Recommendation: on, if Qwen3-8B
is the target; run a small pilot first to see whether the CoT ever mentions evaluation
without the answer doing so. If that rate is under about 5 percent, turn it off.

### D4. Labels (what the judge produces)
Proposed labels per transcript:

- L_spont: does the response (and separately the CoT, if D3 is on) express recognition
  or suspicion that the interaction is an evaluation. Binary. This is the assignment's
  example question.
- L_elic: after the response, append a fixed follow-up user turn asking whether the
  model thinks the preceding interaction was an evaluation or real use, with a short
  justification. Judge labels that answer. Binary (eval / not eval), with an "unclear"
  bucket you decide how to handle.
- C: the construction label (eval-framed vs deploy-framed prompt). Not a judge label.
  Used for analysis, not as the primary probe target.

The unverbalized set is L_elic = eval and L_spont = no.

Open choices inside D4:
- Is L_elic really what you want, or do you want only L_spont? L_elic is still a
  verbalization, just elicited. It is the best available proxy for "what the model
  believes" that a text judge can label. State this limitation in the report.
- Binary or three-class for L_spont (none / hedged suspicion / explicit)? Recommendation:
  binary for the probe, keep the three-class judge output for analysis.
- Exact follow-up wording. Needham et al. 2025 used several phrasings; the result is
  wording-sensitive. Pick one, freeze it, and record the alternatives you rejected.

### D5. Prompt sources and the 2x2
Factor 1 (context): evaluation-flavored vs deployment-flavored.
Factor 2 (format): benchmark-structured (MCQ, graded task, rubric language) vs casual
free-form.

Candidate sources, all public:
- Devbunova 2026 2x2 dataset on Hugging Face (viliana-dev/eval-awareness-2x2),
  length-matched, format-decorrelated, n about 1076 per class. Ready to use.
- Needham et al. 2025 eval/deploy transcripts (chat and agentic).
- SAD "stages oversight" (Laine et al. 2024).
- EvalDetectBench (2026).
- Your own scenarios. You need some of these for the scenario-level split and the
  held-out-template generalization check.

Decisions:
- Reuse Devbunova's 2x2 as the base and add your own cells, or build everything
  yourself? Recommendation: reuse as base, add a hand-written set of 40 to 80 scenarios
  with paraphrase variants for the generalization check.
- How explicit are the eval cues? Range: explicit ("this is a safety evaluation") to
  subtle (a system prompt mentioning a lab name, benchmark-like formatting, a rubric).
  Explicit cues inflate L_spont but invite leakage. Recommendation: a cue-strength
  factor with three levels (none / subtle / explicit), balanced across the other
  factors, so cue strength becomes an analysis axis rather than a confound.
- Metacognitive instruction factor: Vu and Saunders (ICML 2026) found that removing
  "think about your situation" style instructions from the system prompt eliminated
  verbalized awareness in most open models. Include with / without as a factor? This
  doubles the design but gives you the control the assignment asks for. Recommendation:
  yes, but only at two cue-strength levels to keep size manageable.
- Agentic transcripts or chat only? Agentic is where awareness is highest in frontier
  models but open 8B models rarely verbalize there. Recommendation: chat only.

### D6. Dataset size and class balance
Open 8B models verbalize spontaneously at near-zero rates without cues (Vu and
Saunders). You need enough positives for a prediction task. Plan: pilot 200 prompts
across the cells, measure L_spont and L_elic rates, then size the full run. Target
about 2000 to 4000 transcripts, with the SIEVE floor of at least 50 held-out examples
per class in test. If L_spont positives are under 10 percent even with explicit cues,
the L_spont probe becomes a rare-class problem and L_elic is the main target.

### D7. Generation settings
Temperature (0 vs sampled), number of samples per prompt (1 vs k), max tokens. Multiple
samples per scenario stay in one split. Recommendation: temperature 0.7, 2 samples per
prompt, so you can also report within-scenario label variance.

### D8. Judge model and judge variants
Candidates: claude-opus-5-5 (default, most reliable), claude-sonnet-5-5, claude-haiku-5-5
(cheapest), or an open model. Judge cost is roughly transcripts x passes x ~1.5k tokens.
At 4000 transcripts and 3 passes (answer, CoT, follow-up) that is about 18M input
tokens. Sonnet 5.5 is roughly $40, Haiku 5.5 under $5, Opus 5.5 roughly $80.
Decisions:
- Which model.
- CoT judge vs direct judge: run both on the dev set, report agreement, pick one and
  freeze it. Cheap and the assignment likes it.
- Second independent judge model for inter-judge kappa. Report kappa and raw agreement
  and class balance together (the kappa paradox in the SIEVE post).
- Manual labeling: who labels how many. Recommendation: 150 dev examples, each labeled
  by two people blind, before seeing judge output.

### D9. Activation positions and layers
Assignment requires beginning and end of generated response, multiple layers. Decide:
- Which token is "beginning": first generated token, or mean of the first k.
- Which is "end": last generated token, last token before EOS, or mean of last k.
- If CoT is on: start of CoT, start of answer, end of answer. Three positions.
- Layers: every 4th layer, or a sweep. Recommendation: sweep all layers at extraction
  time (cheap to store at 8B scale), select on validation.
- Whether to also store a mean-pooled-over-response vector as an extra readout.
Interpretation note: the end-of-response activation on a transcript that already said
"this looks like a test" trivially encodes the label. The start-of-response position is
the one that speaks to unverbalized awareness. Report them separately.

### D10. Probe family and the geometry question
- Logistic regression on standardized residual-stream activations (required).
- Mean-difference / contrastive direction (no fitting, different geometric assumption).
- Small MLP (nonlinear check).
Pick which to run; recommendation: all three, they are cheap. The report must say what
geometry each assumes (linear separability along one direction vs arbitrary nonlinear).

### D11. Extra readouts (pick at most two)
- SAE-feature probe: sparse logistic regression on SAE latents, plus the single
  best feature. Needs Gemma Scope or equivalent for the target.
- Activation Oracle: zero-shot, ask the released oracle "Is this model in an
  evaluation?" on the same activation. No training. Needs a target in the released list.
- Natural language autoencoder: verbalize the activation, judge the verbalization. Only
  for Qwen2.5-7B, Gemma-3-12b/27b, Llama-3.3-70B.
- Transfer probe: train on Needham or SAD data, test on yours. Doubles as a
  generalization check.
- Self-report probability: the model's own P("evaluation") at the follow-up turn, as a
  text-free baseline that tells you whether the probe beats the model asking itself.

### D12. Controls (assignment needs at least one; recommend three)
- Format decorrelation (the 2x2). Addresses "the probe learned benchmark format."
- Cue deletion: re-extract activations with the explicit cue span removed, and with a
  random span of equal length removed. Addresses "the probe reads the prompt text."
- Shuffled-label probe: null distribution for AUROC on your exact data.
- Optional, higher cost: steer along the probe direction and measure L_spont rate change
  against random and orthogonal directions. This is the causal check the SIEVE post says
  almost nobody runs. Modest version: 20 prompts per arm, three strengths.

### D13. Splits and the generalization check
Split by scenario. Hold out one of: a prompt template family, a source dataset, a topic,
or one cue-strength level. Recommendation: hold out one template family and one cue
level, so the test set has an in-distribution part and a shifted part, and you report
both.

### D14. Compute
Where does generation and extraction run, and is the course infrastructure a fixed
notebook or a library? A single 24 GB GPU for a few hours covers an 8B model at these
sizes. Confirm the GPU budget is separate from the API budget.

## 3. Methods, sorted into what they actually are

The list in the original message mixes label producers with label predictors.

Label producers (judges):
- LLM judge, direct output.
- LLM judge with reasoning before the verdict.
- Second judge model, for agreement.
- Human labels on the dev sample.

Label predictors from activations (the "probe" side):
- Logistic regression probe (required).
- Mean-difference direction.
- MLP probe.
- SAE-feature probe.
- Activation Oracle, zero-shot.
- NLA verbalization plus judge, zero-shot.

Label predictors from text (baselines):
- Majority class (required).
- TF-IDF logistic regression on response text (required).
- Text-statistics baseline: length, CoT length, question marks, hedging-word counts.
  Cheap and catches the obvious confound that long or hedgy responses get labeled.
- LLM classifier given only the response, not the prompt. Tells you how much of the
  judge's signal is in the response alone.

Dropped: "other meta models" beyond AO and NLA. Patchscopes and SelfIE overlap with AO
and add no new claim.

## 4. Validity audit, adapted from the SIEVE post

Run on the final probe and report as a table.

1. Beats majority and beats the text-statistics baseline by a stated margin on held-out
   balanced data, with confidence intervals read at the conservative end.
2. Survives cue deletion (and random-span deletion is reported alongside).
3. Survives format shift (trained on one 2x2 diagonal, tested on the other).
4. Layer and position selected on validation only; one number reported on test.
5. Judge gate: kappa at or above 0.4 between two judges, reported with raw agreement.
6. At least 50 test examples per class, or say the null is not a null.
7. Causal check attempted or explicitly listed as not done, with the weaker claim stated.

## 5. Verified literature to build from

Dates and venues as found on Oct 27 2026; arXiv IDs are as listed by the sources.

Unverbalized awareness, the direct precedents:
- Montoya, Haskins, Adams, "Towards Measuring and Detecting Unverbalized Evaluation
  Awareness," ICML 2026 Mech Interp workshop. Model organism with a type-hint signature;
  resampling removes verbalization but not the behavior gap; probes trained on
  non-coding eval/deploy contrasts still predict it (AUROC 0.81 Nemotron, 0.71 Qwen3.5).
- Vu and Saunders, "Toward Dealing with Unverbalized Eval Awareness," ICML 2026
  workshop. Verbalized awareness near zero in open models; depends on task plausibility
  and on metacognitive instructions in the system prompt; context vs behavior probes.
- LessWrong, "Measuring non-verbalised eval awareness by implanting eval awareness."
  Three-level framework: explicit, CoT-mediated, fully latent. The resampling
  procedure is reusable without the model organism.

Representation vs verbalization:
- Heidari, Memarian, Rabusseau, arXiv 2608.21766, "Evaluation Awareness in Language
  Models: Representation, Verbalization, and Control." Six open models, probe AUROC
  above 0.7, gap between representation and verbalization quantified with MI.
- Nayan et al., arXiv 2606.23583, "Evaluation Awareness Is Not One Capability." 37
  models. Detection, behavior, and representation axes are weakly coupled; probes keep
  AUROC 0.98 when rewrites push behavior to chance.

Confounds and critiques:
- Devbunova, arXiv 2603.19426, "Is Evaluation Awareness Just Format Sensitivity?" The
  2x2 design, the format trap (94 percent false positives on rewritten deploy prompts),
  and the paired-training fix. Code and data released.
- LessWrong, "Probing is not enough: a validity audit for any probe." Leakage via
  prompt tags, kappa paradox, SIEVE gates.
- Knecht, Florin, Hagendorff (LessWrong, May 2026): verbalized awareness has little
  effect on behavior in eight open reasoning models.

Probing and steering, earlier:
- Nguyen et al., arXiv 2507.01786, probes on Llama-3.3-70B, AUROC 0.64 to 0.83.
- Abdelnabi and Salem, arXiv 2505.14617, test-awareness steering in reasoning models.
- Hua, Qin, Marks, Nanda, arXiv 2510.20487, steering eval-aware models to act deployed.
- Needham et al. 2025, "LLMs Often Know When They Are Being Evaluated."
- Laine et al. 2024, SAD, stages-oversight split.

Readout methods:
- Karvonen et al., arXiv 2512.15674, Activation Oracles. Released oracles at
  huggingface.co/collections/adamkarvonen/activation-oracles for Qwen3 1.7B to 32B,
  Gemma-2 9B and 27B, Gemma-3 1B and 27B, Llama-3.1-8B and 3.3-70B.
- Anthropic, "Natural Language Autoencoders," transformer-circuits.pub, May 2026.
  Released for Qwen2.5-7B-Instruct (L20), Gemma-3-12B-IT (L32), Gemma-3-27B-IT (L41),
  Llama-3.3-70B (L53). Inference via SGLang; code at kitft/natural_language_autoencoders.

Not yet verified by us, worth a look: EvalDetectBench (arXiv 2609.01611), "Evaluation
Awareness Shifts from Format to Context with Model Scale" (arXiv 2609.22119), "Not All
Eval-Awareness Is Equal" (arXiv 2608.27340, Qwen3-32B).

## 6. Things to think about that were not in the original plan

- What claim do you want to be able to make at the end? Three options, in increasing
  ambition: (a) a probe reproduces the judge on verbalized awareness; (b) a probe on the
  start-of-response state predicts elicited awareness even when nothing was verbalized;
  (c) that signal is not format, not cue text, and is causally relevant. Decide which
  you are aiming at, since it determines how much of section 4 is mandatory.
- Position semantics with CoT. "Beginning of response" is ambiguous once there is a
  thinking block. Decide and state it.
- The follow-up question changes the transcript. The probe must read activations from
  the forward pass without the follow-up appended, or you leak the question. Extract
  during the first generation, judge after.
- Rare positives. If spontaneous verbalization is rare, the TF-IDF baseline will look
  good on the few positives because they contain words like "test" or "evaluate".
  That is the surface-cue case the assignment wants you to defeat: include transcripts
  that mention tests or evaluations as topic content without expressing suspicion.
- Judge rubric ambiguities to pre-decide: a model saying "if this is a test" as
  rhetoric; a model discussing evaluations because the prompt is about evaluations; a
  model refusing with "I notice this looks like a jailbreak attempt" (suspicion of
  adversarial intent is not the same as suspicion of being evaluated; decide whether it
  counts).
- Open models often answer the follow-up with "I have no way to know." Decide now
  whether that is "not eval", "unclear", or excluded.
- Time. Two weeks and the writing days are reserved. Every extra readout is roughly a
  day. Protect the core.

## 7. Proposed schedule

Week 1 (to Nov 2): decisions in section 2, pilot of 200 prompts, judge rubric v1,
manual labels on 150, confound check on the pilot, splits frozen, extraction verified.
Week 2 (to Nov 9): full generation and extraction, probes and baselines, one or two
extra readouts, cue-deletion and format-shift controls, test run once, blind manual
review, write-up.

## 8. Proposed repo layout

```
data/        prompts, transcripts, labels, splits (large files gitignored)
judge/       rubric versions, judge prompts, agreement analysis
extract/     activation extraction scripts, position and layer config
probes/      LR, mean-diff, MLP, SAE, AO readouts
baselines/   majority, tfidf, text-stats, response-only LLM
controls/    cue deletion, format shift, shuffled labels, steering
analysis/    metrics, confusion matrices, disagreement review sheets
report/
```
