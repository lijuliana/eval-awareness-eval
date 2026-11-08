# Activation readers

Reader code for the hw2 pipeline (github.com/itewari7/hw2). Files go at the same paths inside hw2 (`src/`, `prompts/`, `results/`); they import hw2's `common.py` and `labels_common.py`.

- `src/37_reader_activations.py`: recomputes block activations at each cut point (teacher-forced), with a log-prob check against the stored generations.
- `src/38_activation_readers.py`: runs transfer probe, template lens, logit lens, J-lens, J++, oracle lens, NLA, activation oracle.
- `src/39_reader_tool_checks.py`: sanity checks for each reader; results in `results/reader_tool_checks.json`.
- `src/40_readout_judge.py`: Sonnet judge that scores the word-producing readers, prompt in `prompts/readout_judge_v2.md` (v1 kept for reference).
- `src/41_readout_label_sheet.py`: blind hand-label sheet for the readout judge check.
- `src/42_reader_analysis.py`: frozen analysis (each reader against q, H3, H4, level 3) with hw2's test gate; run once on test.
- `decisions.md`: D-174 to D-182 and D-185 to D-187 (D-183 and D-184 are Ishaan's, in hw2). Scripts were renumbered from 32-36 to 37-41 to avoid clashing with hw2's src/32 and src/33.

- `analysis/`: validation metrics and Figure 1 of the top-level README (`reader_metrics.py`, `text_vs_activation.py`, `fig_main.py`; each takes the path of an hw2 checkout and reads only validation files).

Reader outputs and per-item scores are not in this repo because it is public.
