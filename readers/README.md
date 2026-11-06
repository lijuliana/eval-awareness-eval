# Activation readers

Reader code for the hw2 pipeline (github.com/itewari7/hw2). Files go at the same paths inside hw2 (`src/`, `prompts/`, `results/`); they import hw2's `common.py` and `labels_common.py`.

- `src/32_reader_activations.py`: recomputes block activations at each cut point (teacher-forced), with a log-prob check against the stored generations.
- `src/33_activation_readers.py`: runs transfer probe, template lens, logit lens, J-lens, J++, oracle lens, NLA, activation oracle.
- `src/34_reader_tool_checks.py`: sanity checks for each reader; results in `results/reader_tool_checks.json`.
- `src/35_readout_judge.py`: Sonnet judge that scores the word-producing readers, prompt in `prompts/readout_judge_v2.md` (v1 kept for reference).
- `decisions.md`: D-172 to D-178.

- `analysis/`: validation metrics and Figure 1 of the top-level README (`reader_metrics.py`, `text_vs_activation.py`, `fig_main.py`; each takes the path of an hw2 checkout and reads only validation files).

Reader outputs and per-item scores are not in this repo because it is public.
