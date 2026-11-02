# Hypotheses and frozen choices

Written before any data. Changes are logged at the bottom with a date.

## Hypotheses

- **E1, latent awareness in silence.** Silent traces carry latent awareness. At a given cut point, q on traces that never
  verbalize is above zero and, within a prompt, comparable to q on traces that verbalize later.
  If q on silent traces is near zero, silence is informative and the project reports that.
- **H2, activations beat text.** At least one activation judge predicts q on held-out scenarios better than the strongest
  text reader given the same prompt and prefix. Expected from Montoya et al. 2026 (probe AUROC
  0.71 to 0.81 on suppressed sections).
- **H3, judges agree.** The trained probe and the verbalizing readers (NLA, oracle lens, activation oracle)
  agree with each other about as well as each agrees with q. Expected from Bricknell 2026.
- **H4, single-token lenses lag.** Single-token lenses (J-lens, logit lens) are below the verbalizing readers on this task,
  because "being evaluated" is a multi-token concept. Expected from WorkspaceBench.

## How results will be read

| Result | Reading |
|---|---|
| q(silent) near q(later-verbalizing) within prompt | Whether awareness gets voiced is undetermined at that point |
| q(silent) near zero | Little latent awareness; silence is informative |
| Activation judge beats text reader, but not within prompt | It reads the cue, not the trajectory |
| Activation judge beats text reader, but not after partialling out refusal | It reads refusal intent |
| Activation judge beats text reader, survives both | The activation encodes latent awareness beyond the words |
| Verbalizing readers claim evaluation on real-entity prompts with q = 0 | Confabulation; reported as a false-positive rate |

## Frozen before data

- Splits by near-duplicate cluster; two harm categories held out; committed in `data/splits.json`.
- The 70 resampling scenarios, drawn stratified by split and category, committed.
- Cut offsets, continuations per cut point, continuation cap, sampling settings.
- Judge rubric, after at most three revisions on pilot data only, hashed.
- Readout-judge prompt and the J-lens eval-token set.
- Layers for each judge; probe layer selected on validation only.
- **Test split is not opened until all of the above are frozen.**

## What we do not claim

- That the model believes it is being tested. q is a disposition to voice awareness.
- That any representation causes behavior. No steering is run.
- Transfer beyond this model, this dataset family, and single-turn prompts.

## Change log

- 2026-11-02: first version.
- 2026-11-02: target Qwen3.6-27B; oracle lens added; hedged suspicion counts as verbalized;
  TwinTempt used read-only, without the shortcut-taking anchor.
- 2026-11-02: removed H1 (does the placeholder cue raise the verbalization rate). It was a
  question about verbalization for its own sake. Whether the cue produces enough verbalization
  to resample from is checked in the pilot instead.
