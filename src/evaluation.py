"""
Modül 4 - Değerlendirme metrikleri, görselleştirme ve LLM-as-a-Judge (gpt-4o).

Aşamalar (stage=...):
  generate : 11 modelin D_test-250 (no_robots, D_val-250'den ayrık) istemlerinde tamamlamaları üretilir;
             ortalama yanıt uzunluğu (simge) ve referans SFT modelinden simge bazlı D_KL(pi || pi_ref) hesaplanır
             (hizalanmış modeller: kendi kaynak SFT modeli; SFT modelleri: yalnızca LoRA'sız taban model).
  plots    : SFT/GRPO/DPO/R-DPO kayıp eğrileri, uzunluk, D_KL, tepe VRAM ve süre grafikleri + özet CSV.
  judge    : gpt-4o ile körleştirilmiş, çift yönlü sıralı (A,B)+(B,A) ikili karşılaştırma; konum yanlılığı
             ölçümü ve Uzunluktan Arındırılmış Kazanma Oranı (Length-Controlled Win-Rate %) matrisi
             (istem düzeyinde bootstrap %95 güven aralıklarıyla).
             OPENAI_API_KEY ortam değişkeni gerekir. Yanıtlar diske önbelleklenir; kesilirse kaldığı yerden sürer.
  all      : generate + plots + judge

Kullanım (defter içinden):
    from evaluation import run_evaluation
    out = run_evaluation(stage="judge", judge_n=250)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from tqdm.auto import tqdm

from common import (
    ALIGNED_MODELS,
    ALL_MODELS,
    FIG_DIR,
    GEN_DIR,
    JUDGE_DIR,
    JUDGE_MODEL,
    METRIC_DIR,
    MODEL_DIR,
    SEED,
    SFT_MODELS,
    TARGET_MODEL,
    free_vram,
    has_cuda,
    load_causal_lm,
    load_json,
    load_no_robots_splits,
    load_tokenizer,
    parse_overrides,
    save_json,
    set_seed,
    track_cost,
)

BASELINE = "Model_SFT_IFD1k"
PROPOSED_PAIRS = [  # doğrudan baş-başa (hakem) karşılaştırmalar
    ("Model_SFT_GP_Strat1k", "Model_SFT_IFD1k"),
    ("Model_SFT_GP_Strat1k", "Model_SFT_GP_Top1k"),
    ("Model_SFT_GP_Strat1k", "Model_SFT_Full"),
    ("Model_GP_GRPO", "Model_IFD_GRPO"),
    ("Model_GP_DPO", "Model_IFD_DPO"),
    ("Model_GP_RDPO", "Model_IFD_RDPO"),
    ("Model_GP_RDPO", "Model_GP_DPO"),
    ("Model_IFD_RDPO", "Model_IFD_DPO"),
]


def get_args(argv: list[str] | None = None, **overrides):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "plots", "judge", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Test istemi sayısını sınırla (duman testi)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Var olan tamamlamaları yeniden üret")
    ap.add_argument("--pairs", choices=["key", "all"], default="key",
                    help="key: baseline + önerilen baş-başa çiftler; all: tüm model çiftleri")
    ap.add_argument("--judge-n", type=int, default=0, help="Hakemin göreceği istem sayısı (0 = hepsi)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=300, help="Güven aralığı için bootstrap yeniden örnekleme sayısı")
    ap.add_argument("--dry-run", action="store_true", help="Judge: API çağırmadan yalnızca çağrı sayısını yazdır")
    return parse_overrides(ap, {k.replace("-", "_"): v for k, v in overrides.items()}, argv)


# --------------------------------------------------------------------------- #
# Test kümesi ve adaptör yönetimi
# --------------------------------------------------------------------------- #
def load_test(args) -> pd.DataFrame:
    _, test = load_no_robots_splits()
    return test.iloc[: args.limit].reset_index(drop=True) if args.limit else test


def available_models(args) -> list[str]:
    names = [n for n in ALL_MODELS if (MODEL_DIR / n / "adapter_config.json").exists()]
    if args.models:
        names = [n for n in names if n in args.models]
    return names


class AdapterPool:
    """4-bit 3B taban modeli bir kez yükler; adaptörleri ihtiyaç halinde takıp çıkarır."""

    def __init__(self):
        self.base = load_causal_lm(TARGET_MODEL, quantize=True)
        self.model = None
        self.loaded: set[str] = set()

    def ensure(self, name: str) -> str:
        from peft import PeftModel

        if name in self.loaded:
            return name
        path = MODEL_DIR / name
        if self.model is None:
            self.model = PeftModel.from_pretrained(self.base, path, adapter_name=name)
            self.model.eval()
        else:
            self.model.load_adapter(path, adapter_name=name)
        self.loaded.add(name)
        return name

    def drop(self, name: str, keep: set[str]) -> None:
        if name in self.loaded and name not in keep and len(self.loaded) > 1:
            self.model.delete_adapter(name)
            self.loaded.discard(name)

    def use(self, name: str):
        self.ensure(name)
        self.model.set_adapter(name)
        return self.model


# --------------------------------------------------------------------------- #
# Üretim ve D_KL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_responses(model, tok, prompts: list[str], max_new: int, batch: int) -> list[list[int]]:
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    order = np.argsort([-len(p) for p in prompts])
    out: list[list[int]] = [[] for _ in prompts]
    for s in tqdm(range(0, len(prompts), batch), desc="üretim", leave=False):
        idx = order[s : s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=768, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, i in zip(gen[:, enc["input_ids"].shape[1]:].tolist(), idx):
            ids = []
            for t in row:
                if t == tok.eos_token_id or t == tok.pad_token_id:
                    break
                ids.append(t)
            out[i] = ids
    return out


def _logits_on_response(model, prompt_ids: list[int], resp_ids: list[int], dev) -> torch.Tensor:
    ids = torch.tensor([prompt_ids + resp_ids], device=dev)
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(resp_ids)]
    return logits.float()


@torch.no_grad()
def token_level_kl(pool: AdapterPool, name: str, ref: str | None, tok, prompts, resp_ids) -> float:
    """Ortalama simge bazlı D_KL(pi_name || pi_ref), pi_name'in kendi ürettiği yanıt simgeleri üzerinde.
    ref=None ise LoRA kapatılarak taban model referans alınır."""
    dev = next(pool.base.parameters()).device
    total, count = 0.0, 0
    for p, r in zip(prompts, resp_ids):
        if not r:
            continue
        p_ids = tok(p, add_special_tokens=False)["input_ids"][-768:]
        m = pool.use(name)
        lp_a = F.log_softmax(_logits_on_response(m, p_ids, r, dev), dim=-1)
        if ref is None:
            with m.disable_adapter():
                lp_r = F.log_softmax(_logits_on_response(m, p_ids, r, dev), dim=-1)
        else:
            m = pool.use(ref)
            lp_r = F.log_softmax(_logits_on_response(m, p_ids, r, dev), dim=-1)
        kl = (lp_a.exp() * (lp_a - lp_r)).sum(-1)
        total += kl.sum().item()
        count += kl.numel()
    return total / max(1, count)


def stage_generate(args) -> None:
    set_seed()
    test = load_test(args)
    names = available_models(args)
    if not names:
        print("[uyarı] Değerlendirilecek adaptör yok (outputs/models boş).")
        return
    tok = load_tokenizer(TARGET_MODEL, padding_side="left")
    prompts = test["prompt"].tolist()
    metrics_path = METRIC_DIR / "eval_metrics.json"
    metrics = load_json(metrics_path, {})
    pool = AdapterPool()
    try:
        for name in names:
            gen_path = GEN_DIR / f"{name}.jsonl"
            ref = ALIGNED_MODELS[name][0] if name in ALIGNED_MODELS else None
            if gen_path.exists() and name in metrics and not args.force:
                print(f"[atla] {name}")
                continue
            print(f"\n##### {name} değerlendiriliyor (KL referansı: {ref or 'taban model'})")
            if ref:
                if not (MODEL_DIR / ref / "adapter_config.json").exists():
                    print(f"  [uyarı] referans {ref} yok -> KL taban modele göre hesaplanacak")
                    ref = None
            with track_cost(name, "generation") as rec:
                model = pool.use(name)
                resp_ids = generate_responses(model, tok, prompts, args.max_new_tokens, args.gen_batch)
                kl = token_level_kl(pool, name, ref, tok, prompts, resp_ids)
            texts = tok.batch_decode(resp_ids, skip_special_tokens=True)
            lens = np.array([len(r) for r in resp_ids])
            with open(gen_path, "w", encoding="utf-8") as f:
                for i, row in test.iterrows():
                    f.write(json.dumps({"prompt_id": row["prompt_id"], "category": row["category"],
                                        "response": texts[i].strip(), "n_tokens": int(lens[i])}, ensure_ascii=False) + "\n")
            metrics[name] = {
                "mean_len_tokens": float(lens.mean()), "median_len_tokens": float(np.median(lens)),
                "frac_hit_max_len": float((lens >= args.max_new_tokens).mean()),
                "kl_to_ref": kl, "kl_ref": ref or "base", "n_prompts": len(prompts),
                "gen_seconds": rec.get("seconds"), "gen_peak_vram_gb": rec.get("peak_vram_gb"),
            }
            save_json(metrics, metrics_path)
            print(f"  ort. uzunluk={lens.mean():.1f} simge   D_KL={kl:.4f}")
            for other in list(pool.loaded):
                pool.drop(other, keep={name, ref} if ref else {name})
            free_vram()
    finally:
        del pool
        free_vram()


# --------------------------------------------------------------------------- #
# Grafikler
# --------------------------------------------------------------------------- #
def _series(log: list[dict], key: str):
    pts = [(r["step"], r[key]) for r in log if key in r and "step" in r and np.isfinite(r[key])]
    return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])


def _smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def stage_plots(args) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {n: load_json(METRIC_DIR / f"{n}_log.json", []) for n in ALL_MODELS}
    logs = {n: l for n, l in logs.items() if l}
    groups = {
        "SFT kaybı": [n for n in SFT_MODELS if n in logs],
        "DPO kaybı": [n for n in ALIGNED_MODELS if n.endswith("_DPO") and n in logs],
        "R-DPO kaybı": [n for n in ALIGNED_MODELS if n.endswith("_RDPO") and n in logs],
        "GRPO kaybı": [n for n in ALIGNED_MODELS if n.endswith("_GRPO") and n in logs],
    }
    groups = {k: v for k, v in groups.items() if v}
    if groups:
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
        for ax, (title, names) in zip(axes[0], groups.items()):
            for n in names:
                x, y = _series(logs[n], "loss")
                if x:
                    ax.plot(x, _smooth(y), label=n.replace("Model_", ""))
            ax.set_title(title)
            ax.set_xlabel("adım")
            ax.set_ylabel("kayıp")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "loss_curves.png", dpi=150)
        plt.close()

    grpo = [n for n in groups.get("GRPO kaybı", [])]
    dpo_like = groups.get("DPO kaybı", []) + groups.get("R-DPO kaybı", [])
    if grpo or dpo_like:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for n in grpo:
            x, y = _series(logs[n], "reward")
            if x:
                axes[0].plot(x, _smooth(y), label=n.replace("Model_", ""))
        axes[0].set(title="GRPO ortalama ödül r_phi", xlabel="adım", ylabel="ödül")
        for n in dpo_like:
            x, y = _series(logs[n], "rewards/margins")
            if x:
                axes[1].plot(x, _smooth(y), label=n.replace("Model_", ""))
        axes[1].set(title="DPO/R-DPO örtük ödül marjı", xlabel="adım", ylabel="marj")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "reward_curves.png", dpi=150)
        plt.close()

    metrics = load_json(METRIC_DIR / "eval_metrics.json", {})
    costs = load_json(METRIC_DIR / "training_costs.json", {})
    rows = []
    for n in ALL_MODELS:
        stage = "sft" if n in SFT_MODELS else ALIGNED_MODELS[n][1]
        c = costs.get(f"{stage}:{n}", {})
        m = metrics.get(n, {})
        loss_x, loss_y = _series(logs.get(n, []), "loss")
        rows.append({
            "model": n, "stage": stage, "train_seconds": c.get("seconds"), "train_peak_vram_gb": c.get("peak_vram_gb"),
            "final_train_loss": loss_y[-1] if loss_y else None, "mean_len_tokens": m.get("mean_len_tokens"),
            "median_len_tokens": m.get("median_len_tokens"), "kl_to_ref": m.get("kl_to_ref"), "kl_ref": m.get("kl_ref"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRIC_DIR / "summary_table.csv", index=False)
    d = df.dropna(subset=["mean_len_tokens"])
    if len(d):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(d["model"].str.replace("Model_", ""), d["mean_len_tokens"], color="tab:blue")
        axes[0].set(title="Ortalama yanıt uzunluğu (simge)", ylabel="simge")
        dk = d.dropna(subset=["kl_to_ref"])
        axes[1].bar(dk["model"].str.replace("Model_", ""), dk["kl_to_ref"], color="tab:red")
        axes[1].set(title="Simge bazlı D_KL (aligned→kaynak SFT; SFT→taban)", ylabel="nat/simge")
        for ax in axes:
            ax.tick_params(axis="x", rotation=70)
            ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "length_and_kl.png", dpi=150)
        plt.close()

    dc = df.dropna(subset=["train_seconds"])
    if len(dc):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(dc["model"].str.replace("Model_", ""), dc["train_peak_vram_gb"], color="tab:green")
        axes[0].set(title="Eğitimde en yüksek GPU VRAM (GB)", ylabel="GB")
        axes[1].bar(dc["model"].str.replace("Model_", ""), dc["train_seconds"] / 60, color="tab:orange")
        axes[1].set(title="Eğitim süresi (dk)", ylabel="dakika")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.tick_params(axis="x", rotation=70)
            ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cost.png", dpi=150)
        plt.close()
    print(f"[grafikler] {FIG_DIR}")
    return df


# --------------------------------------------------------------------------- #
# LLM-as-a-Judge
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are an impartial, expert evaluator of AI assistant responses. You will see a conversation context and two "
    "candidate responses (A and B) to its last user turn. Decide which response is better overall: helpfulness, "
    "correctness, instruction following, relevance, and clarity. Do NOT prefer a response because of its length, "
    "its position (A or B), or its formatting; a concise correct answer beats a verbose one. If they are of truly "
    'equal quality answer "tie". Reply with strict JSON only: {"reasoning": "<max 2 short sentences>", '
    '"winner": "A" | "B" | "tie"}.'
)


def judge_user_prompt(context: str, a: str, b: str) -> str:
    return (f"### Conversation context\n{context[:6000]}\n\n### Response A\n{a[:6000]}\n\n"
            f"### Response B\n{b[:6000]}\n\nWhich response is better? Answer with the JSON only.")


def load_generations(names: list[str]) -> dict[str, dict[str, dict]]:
    gens = {}
    for n in names:
        p = GEN_DIR / f"{n}.jsonl"
        if p.exists():
            gens[n] = {r["prompt_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}
    return gens


def build_pairs(names: list[str], mode: str) -> list[tuple[str, str]]:
    have = set(names)
    if mode == "all":
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pairs = [(m, BASELINE) for m in names if m != BASELINE]
    pairs += [p for p in PROPOSED_PAIRS if p[0] in have and p[1] in have]
    seen, out = set(), []
    for a, b in pairs:
        key = frozenset((a, b))
        if a in have and b in have and key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


class JudgeClient:
    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def __call__(self, context: str, a: str, b: str) -> str:
        delay = 2.0
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=150,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": judge_user_prompt(context, a, b)}],
                )
                w = str(json.loads(r.choices[0].message.content).get("winner", "")).strip().lower()
                return {"a": "A", "b": "B"}.get(w, "tie")
            except Exception as e:  # hız sınırı / ağ hatası -> üstel geri çekilme
                if attempt == 7:
                    raise
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        return "tie"


def lc_winrate(df: pd.DataFrame) -> tuple[float, float]:
    """
    Uzunluktan arındırılmış kazanma oranı (AlpacaEval-LC benzeri): P(kazan) = sigmoid(b + w * tanh(dLen/std)).
    Karar skorları (kazan=1, beraberlik=0.5, kayıp=0) ağırlıklı çift etiket olarak uydurulur; LC = sigmoid(b),
    yani uzunluk farkı sıfırken beklenen kazanma olasılığı. Dönüş: (ham %, LC %).
    """
    from sklearn.linear_model import LogisticRegression

    score = df["score"].to_numpy(dtype=float)
    raw = 100 * score.mean()
    dlen = df["dlen"].to_numpy(dtype=float)
    sd = dlen.std()
    if len(df) < 4 or sd < 1e-9 or score.min() == score.max():
        return raw, raw
    x = np.tanh(dlen / sd).reshape(-1, 1)
    X = np.vstack([x, x])
    y = np.r_[np.ones(len(x)), np.zeros(len(x))]
    w = np.r_[score, 1.0 - score]
    clf = LogisticRegression(C=100.0, fit_intercept=True).fit(X, y, sample_weight=np.clip(w, 1e-9, None))
    return raw, 100.0 / (1.0 + np.exp(-clf.intercept_[0]))


def bootstrap_winrate_ci(df: pd.DataFrame, n_boot: int = 300, alpha: float = 0.05, seed: int = SEED) -> dict:
    """İstem (prompt_id) düzeyinde bootstrap ile ham ve LC kazanma oranı için %95 güven aralığı."""
    rng = np.random.RandomState(seed)
    groups = [g for _, g in df.groupby("pid")]
    raws, lcs = [], []
    for _ in range(n_boot):
        boot = pd.concat([groups[i] for i in rng.randint(0, len(groups), len(groups))], ignore_index=True)
        r, l = lc_winrate(boot)
        raws.append(r)
        lcs.append(l)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"raw_lo": np.percentile(raws, q[0]), "raw_hi": np.percentile(raws, q[1]),
            "lc_lo": np.percentile(lcs, q[0]), "lc_hi": np.percentile(lcs, q[1])}


def stage_judge(args) -> dict:
    names = available_models(args)
    gens = load_generations(names)
    names = [n for n in names if n in gens]
    pairs = build_pairs(names, args.pairs)
    if not pairs:
        print("[uyarı] Karşılaştırılacak çift yok; önce stage='generate' çalıştırın.")
        return {}
    test = load_test(args)
    if args.judge_n:
        test = test.iloc[: args.judge_n]
    ctx = dict(zip(test["prompt_id"], test["context"]))
    pids = [p for p in test["prompt_id"] if all(p in gens[n] for n in names)]
    jobs = []
    cache_path = JUDGE_DIR / "judgements.jsonl"
    done: dict[str, str] = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["key"]] = r["winner"]
    for a, b in pairs:
        for pid in pids:
            for first, second in ((a, b), (b, a)):  # (A,B) ve (B,A)
                key = f"{first}|{second}|{pid}"
                if key not in done:
                    jobs.append((key, first, second, pid))
    print(f"[hakem] {len(pairs)} çift x {len(pids)} istem x 2 sıra = {len(pairs) * len(pids) * 2} yargı; "
          f"önbellekte {len(done)}, çağrılacak {len(jobs)} ({args.judge_model})")
    if args.dry_run:
        return {"n_calls": len(jobs), "n_cached": len(done)}
    if jobs:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY ayarlanmamış (os.environ['OPENAI_API_KEY'] = '...').")
        judge, lock = JudgeClient(args.judge_model), threading.Lock()
        with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.judge_workers) as ex:
            futs = {ex.submit(judge, ctx[pid], gens[first][pid]["response"], gens[second][pid]["response"]): (key, first, second)
                    for key, first, second, pid in jobs}
            for fu in tqdm(as_completed(futs), total=len(futs), desc="gpt-4o"):
                key, _, _ = futs[fu]
                w = fu.result()
                done[key] = w
                with lock:
                    f.write(json.dumps({"key": key, "winner": w}) + "\n")
                    f.flush()

    # Sonuçları topla
    rows, bias = [], {"A": 0, "B": 0, "tie": 0}
    for a, b in pairs:
        recs = []
        for pid in pids:
            for first, second in ((a, b), (b, a)):
                w = done.get(f"{first}|{second}|{pid}")
                if w is None:
                    continue
                bias[w] += 1
                s = 0.5 if w == "tie" else float((w == "A") == (first == a))
                recs.append({"score": s, "dlen": gens[a][pid]["n_tokens"] - gens[b][pid]["n_tokens"], "order": first == a,
                             "pid": pid, "w": w})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        raw, lc = lc_winrate(df)
        ci = bootstrap_winrate_ci(df, n_boot=args.n_boot)
        flips = df.groupby("pid")["score"].agg(lambda s: abs(s.iloc[0] - s.iloc[-1]) < 1e-9 if len(s) == 2 else True)
        per_prompt = df.groupby("pid")["score"].mean()  # iki sıranın ortalaması -> istem düzeyinde sonuç
        wins, losses = int((per_prompt > 0.5).sum()), int((per_prompt < 0.5).sum())
        p_sign = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
        rows.append({"model": a, "opponent": b, "n": len(df), "raw_winrate": raw, "lc_winrate": lc, **ci,
                     "wins": wins, "losses": losses, "ties": int(len(per_prompt) - wins - losses),
                     "sign_test_p": p_sign, "order_consistency": float(flips.mean())})
    res = pd.DataFrame(rows)
    res.to_csv(JUDGE_DIR / "pairwise_results.csv", index=False)

    models = sorted({*res["model"], *res["opponent"]}, key=ALL_MODELS.index)
    lc_mat = pd.DataFrame(np.nan, index=models, columns=models)
    raw_mat = lc_mat.copy()
    for r in res.itertuples():
        lc_mat.loc[r.model, r.opponent], lc_mat.loc[r.opponent, r.model] = r.lc_winrate, 100 - r.lc_winrate
        raw_mat.loc[r.model, r.opponent], raw_mat.loc[r.opponent, r.model] = r.raw_winrate, 100 - r.raw_winrate
    lc_mat.to_csv(JUDGE_DIR / "lc_winrate_matrix.csv")
    raw_mat.to_csv(JUDGE_DIR / "raw_winrate_matrix.csv")
    tot = max(1, sum(bias.values()))
    print(f"\n[konum yanlılığı] A seçimi={bias['A'] / tot:.1%}  B seçimi={bias['B'] / tot:.1%}  beraberlik={bias['tie'] / tot:.1%}")
    save_json({"position_bias": bias, "judge_model": args.judge_model}, JUDGE_DIR / "judge_summary.json")
    plot_matrix(lc_mat)
    return {"pairwise": res, "lc_matrix": lc_mat, "raw_matrix": raw_mat, "position_bias": bias}


def plot_matrix(mat: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m.replace("Model_", "") for m in mat.index]
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 0.9 * len(labels) + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("gpt-4o Length-Controlled Win-Rate % (satır > sütun)")
    fig.colorbar(im, ax=ax, fraction=0.035)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lc_winrate_matrix.png", dpi=150)
    plt.close()


def run_evaluation(args=None, **overrides) -> dict:
    """stage = generate | plots | judge | all. Her aşamanın sonucunu bir sözlükte döndürür."""
    args = args or get_args(**overrides)
    out: dict = {}
    if args.stage in ("generate", "all"):
        stage_generate(args)
    if args.stage in ("plots", "all"):
        out["summary"] = stage_plots(args)
    if args.stage in ("judge", "all"):
        out["judge"] = stage_judge(args)
    return out
