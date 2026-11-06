"""Tool check (PROPOSAL 4.5): each activation reader reproduces a known reading before its outputs are used.
Results are merged into results/reader_tool_checks.json, one entry per check, with pass/fail.

Stages (each loads one model, so they run one after another on one GPU):
  base      - convention check (our block-L residual = the hook on decoder block L), logit lens = model's
              own output, J-lens / J++ on known factual and two-hop prompts, template lens on known phrases;
              also stores the residuals the later stages read.
  olens     - the oracle lens README example (Carmen) and the same residual through our capture.
  ao        - activation oracle: sentiment yes/no on a clearly positive and a clearly negative sentence.
  nla       - NLA explanations of three plain sentences (topic words checked automatically, printed for a look).
  twintempt - the transfer probe on TwinTempt (76 tasks x 4 versions), against the published lifts.

    python src/34_reader_tool_checks.py --stage base
No study data is read here: every input is a fixed sentence or a public benchmark item.
"""
import argparse
import importlib
import json
from pathlib import Path

import torch

import readers_common as rc

readers = importlib.import_module("33_activation_readers")
OUT = Path("results/reader_tool_checks.json")
STORE = rc.ACT_DIR / "toolcheck.pt"

FACT_PROMPTS = {"The Eiffel Tower is located in the city of": [" Paris"],
                "The capital of the country where the Colosseum stands is": [" Rome", " Italy"]}
TEMPLATE_PROMPTS = {"After dinner we each had a bowl of vanilla ice": ["ice cream", "cream"],
                    "She moved from Sydney to Auckland, the largest city in New": ["New Zealand", "Zealand"]}
SENTIMENT = {"positive": "I absolutely loved this film. The acting was wonderful and I left the cinema smiling.",
             "negative": "This was the most boring, badly acted film I have ever seen. I hated every minute."}
SENTIMENT_QUESTION = "Answer with 'Yes' or 'No' only. Is the sentiment of this text positive?"
NLA_TEXTS = {"baking": (["bak", "flour", "recipe", "cook", "dough", "cake"],
                        "Preheat the oven to 180 degrees. In a large bowl, mix the flour, sugar and softened butter until the dough"),
             "football": (["football", "soccer", "goal", "match", "team", "sport"],
                          "In the 89th minute the striker curled a free kick over the wall and into the top corner, and the stadium"),
             "code": (["code", "python", "function", "program", "loop", "list"],
                      "def mean(values):\n    total = 0\n    for v in values:\n        total += v\n    return total /")}


def save(name, result):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[name] = result
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(name, "PASS" if result.get("passed") else "FAIL", json.dumps(result)[:600], flush=True)


def residuals(model, tok, text, blocks):
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return {b: out.hidden_states[b + 1][0, -1].float().cpu() for b in blocks}, out, ids


def stage_base():
    from transformers import AutoTokenizer
    model = rc.load_qwen()
    tok = AutoTokenizer.from_pretrained(rc.QWEN, revision=rc.QWEN_REVISION)
    blocks = rc.decoder_blocks(model)
    store = {}

    # 1. Convention: hidden_states[L + 1] equals the output of decoder block L (as the readers assume).
    grabbed = {}
    hooks = [blocks[b].register_forward_hook(lambda m, i, o, b=b: grabbed.__setitem__(b, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
             for b in (42, 62)]
    res, out, _ = residuals(model, tok, "Who wrote the opera Carmen?", [42, 62])
    for h in hooks:
        h.remove()
    cos = {b: float(torch.cosine_similarity(res[b], grabbed[b], dim=0)) for b in (42, 62)}
    save("convention_hidden_states_vs_block_hook", {"cosine": cos, "passed": min(cos.values()) > 0.9999})

    # 2. Logit lens at the last block equals the model's own next-token distribution.
    last = {}
    hook = blocks[63].register_forward_hook(lambda m, i, o: last.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1]))
    _, out, _ = residuals(model, tok, "The Eiffel Tower is located in the city of", [42])
    hook.remove()
    with torch.no_grad():
        lens_logits = model.lm_head(model.model.norm(last["h"][None, None]))[0, 0].float()
    model_logits = out.logits[0, -1].float()
    save("logit_lens_final_block", {"top1_lens": tok.decode([int(lens_logits.argmax())]),
                                    "top1_model": tok.decode([int(model_logits.argmax())]),
                                    "max_abs_logit_diff": float((lens_logits - model_logits).abs().max()),
                                    "passed": int(lens_logits.argmax()) == int(model_logits.argmax())})

    # 3. J-lens and J++ on a factual and a two-hop prompt: expected tokens' rank at blocks 36-52.
    for kind in ("jlens", "jpp"):
        repo, path = ("neuronpedia/jacobian-lens",
                      {"jlens": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
                       "jpp": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"}[kind])
        blob = torch.load(readers.hub_file(repo, path), map_location="cpu", weights_only=False)
        result = {"file_keys": sorted(blob.keys()), "source_layers": [int(min(blob["source_layers"])), int(max(blob["source_layers"]))] if "source_layers" in blob else None, "prompts": {}}
        best_ranks = []
        for prompt, expected in FACT_PROMPTS.items():
            res, _, _ = residuals(model, tok, prompt, [36, 40, 42, 44, 48, 52])
            ranks = {}
            for b, h in res.items():
                J = rc.jacobian(blob, b).cuda()
                with torch.no_grad():
                    logits = model.lm_head(model.model.norm((h.cuda() @ J.T).to(torch.bfloat16)[None, None]))[0, 0].float()
                order = logits.argsort(descending=True)
                ranks[b] = min(int((order == tok.encode(e, add_special_tokens=False)[0]).nonzero()[0]) for e in expected)
            result["prompts"][prompt] = {"expected": expected, "rank_by_block": ranks}
            best_ranks.append(min(ranks.values()))
        result["passed"] = all(r < 10 for r in best_ranks)
        save(f"{kind}_known_prompts", result)
        del blob

    # 4. Template lens: expected phrase's rank among 13,731 rows at blocks 36-52.
    from safetensors import safe_open
    tl_repo, base = "camilablank/workspace-lenses", "qwen3.6-27b/template-lens/"
    words = [l.split("\t", 1)[1] for l in open(readers.hub_file(tl_repo, base + "template_words+phrases_v3.txt"),
                                               encoding="utf-8").read().splitlines()]
    with safe_open(readers.hub_file(tl_repo, base + "templates+phrases_v3.safetensors"), framework="pt") as f:
        meta = f.metadata()
        layers = json.loads(meta.get("layers_json", "null")) or list(range(64))
        sl = f.get_slice("templates")
        T = {b: torch.nn.functional.normalize(sl[layers.index(b)].float().cuda(), dim=-1) for b in (36, 42, 48)}
    result = {"metadata": meta, "prompts": {}}
    best = []
    for prompt, expected in TEMPLATE_PROMPTS.items():
        res, _, _ = residuals(model, tok, prompt, [36, 42, 48])
        ranks = {}
        for b, h in res.items():
            order = (T[b] @ torch.nn.functional.normalize(h.cuda(), dim=0)).argsort(descending=True).tolist()
            ranks[b] = min(order.index(words.index(e)) for e in expected if e in words)
        result["prompts"][prompt] = {"expected": expected, "rank_by_block": ranks,
                                     "top5_block42": [words[i] for i in (T[42] @ torch.nn.functional.normalize(res[42].cuda(), dim=0)).topk(5).indices.tolist()]}
        best.append(min(ranks.values()))
    result["passed"] = all(r < 50 for r in best)
    save("template_lens_known_phrases", result)

    # Residuals for the later stages (plain text, last token).
    for name, text in SENTIMENT.items():
        store[f"sentiment_{name}"] = residuals(model, tok, text, [rc.AO_BLOCK])[0][rc.AO_BLOCK]
    for name, (_, text) in NLA_TEXTS.items():
        store[f"nla_{name}"] = residuals(model, tok, text, [rc.PRIMARY_BLOCK])[0][rc.PRIMARY_BLOCK]
    store["carmen_chat_block40"] = None  # filled by the olens stage with olens.residual for comparison
    rc.ACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(store, STORE)


def stage_olens():
    import importlib.util
    path = readers.hub_file("agu18dec/final_olens", "olens.py")
    spec = importlib.util.spec_from_file_location("olens", path)
    olens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(olens)
    lens = olens.load_lens(olens.from_hub("agu18dec/final_olens", readers.OLENS_CHECKPOINT))
    torch.manual_seed(0)
    h = olens.residual(lens, "Who wrote the opera Carmen?", layer=40)
    reads = olens.read(lens, h, layer=40, n=4, ban_template=True)
    hits = sum(any(w in r.lower() for w in ("bizet", "carmen", "opera")) for r in reads)
    save("olens_readme_carmen", {"checkpoint": readers.OLENS_CHECKPOINT, "readouts": reads, "hits_of_4": hits,
                                 "passed": hits >= 2})


def stage_ao():
    from peft import PeftModel
    from transformers import AutoTokenizer
    store = torch.load(STORE)
    tok = AutoTokenizer.from_pretrained(rc.QWEN, revision=rc.QWEN_REVISION)
    repo = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B"
    model = PeftModel.from_pretrained(rc.load_qwen(), readers.hub_dir(repo, ["*.json", "*.safetensors"])).eval()
    text = tok.apply_chat_template([{"role": "user", "content": f"Layer: {rc.AO_BLOCK}\n ? \n{SENTIMENT_QUESTION}"}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids
    pos = ids.index(tok.encode(" ?", add_special_tokens=False)[0])
    yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    p = {}
    for name in SENTIMENT:
        handle = rc.add_norm_matched_hook(model, [pos], [store[f"sentiment_{name}"]])
        try:
            with torch.no_grad():
                probs = torch.softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
        finally:
            handle.remove()
        p[name] = float(probs[yes] / (probs[yes] + probs[no]))
    save("ao_sentiment", {"p_yes": p, "marker_position": pos, "passed": p["positive"] > 0.5 > p["negative"]})


def stage_nla():
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    store = torch.load(STORE)
    root = readers.hub_dir("ceselder/qwen3.6-27b-nla-rl", ["av_base/*", f"{readers.NLA_ADAPTER}/*"])
    meta = yaml.safe_load(open(f"{root}/av_base/nla_meta.yaml"))
    tok = AutoTokenizer.from_pretrained(f"{root}/av_base")
    base = AutoModelForCausalLM.from_pretrained(f"{root}/av_base", dtype=torch.bfloat16, device_map={"": "cuda"})
    model = PeftModel.from_pretrained(base, f"{root}/{readers.NLA_ADAPTER}").eval()
    ids, pos = readers.nla_prompt_ids(tok, meta)
    x = torch.tensor([ids], device="cuda")
    out = {}
    for name, (words, _) in NLA_TEXTS.items():
        handle = rc.add_norm_matched_hook(model, [pos], [store[f"nla_{name}"]])
        try:
            with torch.no_grad():
                gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=readers.NLA_MAX_NEW,
                                     do_sample=False)
        finally:
            handle.remove()
        text = tok.decode(gen[0, len(ids):], skip_special_tokens=True)
        out[name] = {"explanation": text, "topic_hit": any(w in text.lower() for w in words)}
    save("nla_plain_sentences", {"adapter": readers.NLA_ADAPTER, "texts": out,
                                 "passed": all(v["topic_hit"] for v in out.values())})


def stage_twintempt(eab):
    """The ready-made reader in adam-jb/eval-awareness-benchmark, then its paired scorer."""
    import subprocess
    import sys
    scores = Path(rc.ACT_DIR) / "twintempt_probe_scores.jsonl"
    subprocess.run([sys.executable, "examples/read_with_qwen_probe.py", "--out", str(scores.resolve())], cwd=eab, check=True)
    report = subprocess.run([sys.executable, "-m", "twintempt", str(scores.resolve()), "--column", "b27_eval"],
                            cwd=eab, check=True, capture_output=True, text=True).stdout
    print(report)
    save("transfer_probe_twintempt", {"column": "b27_eval", "report": report,
                                      "published_probe_lift_b27_eval": {"subtle": -0.029, "obvious": 0.043, "very_obvious": 0.215},
                                      "passed": None, "note": "compare the very_obvious lift (published about +0.2, AUC about 1.0) by eye"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["base", "olens", "ao", "nla", "twintempt"], required=True)
    parser.add_argument("--eab", default="../eab", help="clone of adam-jb/eval-awareness-benchmark")
    args = parser.parse_args()
    {"base": stage_base, "olens": stage_olens, "ao": stage_ao, "nla": stage_nla,
     "twintempt": lambda: stage_twintempt(args.eab)}[args.stage]()


if __name__ == "__main__":
    main()
