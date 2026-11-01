"""
Modül 2 - DEAP ile genetik programlama (GP) ve hızlı vekil uygunluk (proxy fitness).

Her GP bireyi T_j, 6 normalize özniteliği (Lx, Ly, PPL, IFD, Hy, Ddiv) bir S_i skoruna dönüştüren
sembolik bir formüldür. Uygunluk değerlendirmesi:
  1. T_j tüm D_havuz için S_i skorlarını vektörize hesaplar.
  2. Tabakalı Top-K: 20 kümenin her birinden en yüksek S_i'li 50 örnek -> 1.000'lik D_aday.
  3. Seçilen indekslerin sıralı tuple'ının SHA-1 özeti `fitness_cache`'te aranır; varsa GPU eğitimi atlanır.
  4. Qwen2.5-0.5B-Base 4-bit QLoRA modeli GPU'ya YALNIZCA BİR KEZ yüklenir; her yeni bireyde LoRA
     ağırlıkları `reset_lora_parameters` ile başlangıç durumuna döndürülür ve 60 adım mikro-SFT yapılır.
  5. Uygunluk = L_val (D_val-250 çapraz entropisi) + 0.001 * |T_j|   (minimize edilir).

Evrim: P=16, G=10, genHalfAndHalf(derinlik 1-4), turnuva k=3, HallOfFame(2) elitizmi,
       cxOnePoint (0.7), mutUniform (0.2).

Çıktılar (outputs/gp): best_formula.json, subsets/*.json, history.csv, fitness_cache.json, evolution.png

Kullanım (defter içinden):
    from gp_evolution import run_gp_evolution
    res = run_gp_evolution(pop=16, gens=10, steps=60)
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import operator
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from deap import algorithms, base, creator, gp, tools

from common import (
    FEAT_DIR,
    FEATURE_NAMES,
    GP_DIR,
    K_SUBSET,
    N_CLUSTERS,
    PER_CLUSTER,
    PROXY_MODEL,
    SEED,
    PadCollator,
    compute_dtype,
    format_prompt,
    free_vram,
    has_cuda,
    load_causal_lm,
    load_json,
    load_no_robots_splits,
    load_tokenizer,
    lora_config,
    parse_overrides,
    save_json,
    set_seed,
    tokenize_pairs,
)

SUBSET_DIR = GP_DIR / "subsets"
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = GP_DIR / "fitness_cache.json"
PROXY_MAX_LEN, PROXY_MAX_PROMPT = 384, 192
PENALTY = 1e3
BLOAT_COEF = 1e-3


# --------------------------------------------------------------------------- #
# Fonksiyon ve terminal kümeleri
# --------------------------------------------------------------------------- #
def protected_div(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    small = np.abs(b) < 1e-6
    return np.where(small, 1.0, a / np.where(small, 1.0, b))


def protected_log(a):
    return np.log(np.abs(a) + 1e-6)


def protected_sqrt(a):
    return np.sqrt(np.abs(a))


def build_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", len(FEATURE_NAMES))
    pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(FEATURE_NAMES)})
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="pdiv")
    pset.addPrimitive(protected_log, 1, name="plog")
    pset.addPrimitive(protected_sqrt, 1, name="psqrt")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    pset.addEphemeralConstant("erc", lambda: round(random.uniform(-1.0, 1.0), 4))
    return pset


def to_infix(ind) -> str:
    """DEAP önek (prefix) ağacını okunabilir infix biçime çevirir."""
    sym = {"add": "+", "sub": "-", "mul": "*", "pdiv": "/"}

    def rec(i: int):
        node = ind[i]
        if isinstance(node, gp.Terminal):
            v = node.value
            return (f"{v:.4g}" if isinstance(v, float) else str(v)), i + 1
        args, j = [], i + 1
        for _ in range(node.arity):
            s, j = rec(j)
            args.append(s)
        if node.name in sym:
            return f"({args[0]} {sym[node.name]} {args[1]})", j
        return f"{node.name}({', '.join(args)})", j

    return rec(0)[0]


# --------------------------------------------------------------------------- #
# Tabakalı seçim
# --------------------------------------------------------------------------- #
def stratified_topk(scores: np.ndarray, members: list[np.ndarray], per_cluster: int, k_total: int) -> np.ndarray:
    chosen = []
    for mem in members:
        order = np.argsort(-scores[mem], kind="stable")[: min(per_cluster, len(mem))]
        chosen.append(mem[order])
    sel = np.concatenate(chosen)
    if len(sel) < k_total:  # küçük kümeler için kalan kotayı genel en yüksek skorlarla doldur
        rest = np.setdiff1d(np.arange(len(scores)), sel)
        sel = np.concatenate([sel, rest[np.argsort(-scores[rest], kind="stable")[: k_total - len(sel)]]])
    return np.sort(sel[:k_total])


def subset_key(sel: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(tuple(sel.tolist()), dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Vekil model: bir kez yüklenir, LoRA her bireyde sıfırlanır
# --------------------------------------------------------------------------- #
class ProxyTrainer:
    def __init__(self, pool: pd.DataFrame, args):
        from peft import get_peft_model, prepare_model_for_kbit_training

        self.args = args
        self.tok = load_tokenizer(PROXY_MODEL)
        self.collate = PadCollator(self.tok.pad_token_id)
        self.dtype = compute_dtype()

        print("[vekil] havuz ve doğrulama kümesi simgeleştiriliyor...")
        self.pool_tok = tokenize_pairs(
            self.tok,
            [format_prompt(p) for p in pool["prompt"]],
            pool["response"].tolist(),
            PROXY_MAX_LEN,
            PROXY_MAX_PROMPT,
        )
        val, _ = load_no_robots_splits()
        self.val_tok = tokenize_pairs(self.tok, val["prompt"].tolist(), val["response"].tolist(), PROXY_MAX_LEN, PROXY_MAX_PROMPT)
        self.val_tok.sort(key=lambda e: len(e["input_ids"]))

        print(f"[vekil] {PROXY_MODEL} 4-bit QLoRA modeli GPU'ya yükleniyor (tek sefer)...")
        model = load_causal_lm(PROXY_MODEL, quantize=has_cuda())
        if has_cuda():
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        self.model = get_peft_model(model, lora_config(r=16, alpha=32, dropout=0.05))
        self.model.print_trainable_parameters()
        self.lora_layers = [m for m in self.model.modules() if hasattr(m, "reset_lora_parameters") and hasattr(m, "lora_A")]
        self.device = next(self.model.parameters()).device
        self.n_trained = 0

    def reset_lora(self) -> None:
        """Her bireyde aynı başlangıç: A ~ Kaiming-uniform, B = 0 (model yeniden yüklenmez)."""
        torch.manual_seed(SEED)
        for layer in self.lora_layers:
            for adapter in list(layer.lora_A.keys()):
                layer.reset_lora_parameters(adapter, True)

    def _autocast(self):
        if has_cuda():
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)

    def train_and_eval(self, idx: np.ndarray) -> float:
        a = self.args
        from transformers import get_cosine_schedule_with_warmup

        self.reset_lora()
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.1 * a.steps)), a.steps)
        scaler = torch.amp.GradScaler("cuda", enabled=has_cuda() and self.dtype == torch.float16)

        rng = np.random.RandomState(SEED)
        order = np.concatenate([rng.permutation(idx) for _ in range(a.steps * a.batch // len(idx) + 2)])
        for step in range(a.steps):
            chunk = order[step * a.batch : (step + 1) * a.batch]
            examples = [self.pool_tok[i] for i in chunk]
            n_tok = max(1, sum(len(e["input_ids"]) - e["prompt_len"] for e in examples))
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(examples), a.micro_batch):
                micro = self.collate(examples[s : s + a.micro_batch])
                micro = {k: v.to(self.device) for k, v in micro.items()}
                n_micro = (micro["labels"][:, 1:] != -100).sum().item()
                with self._autocast():
                    loss = self.model(**micro).loss * (n_micro / n_tok)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        del opt, sched, scaler, params
        val_loss = self.validation_loss()
        self.n_trained += 1
        return val_loss

    @torch.no_grad()
    def validation_loss(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for s in range(0, len(self.val_tok), 8):
            batch = {k: v.to(self.device) for k, v in self.collate(self.val_tok[s : s + 8]).items()}
            with self._autocast():
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            tgt = batch["labels"][:, 1:]
            mask = tgt != -100
            total += F.cross_entropy(logits[:, :-1][mask].float(), tgt[mask], reduction="sum").item()
            count += int(mask.sum().item())
        self.model.train()
        return total / max(1, count)


# --------------------------------------------------------------------------- #
# Uygunluk
# --------------------------------------------------------------------------- #
class FitnessEvaluator:
    def __init__(self, X: np.ndarray, clusters: np.ndarray, pset, trainer: ProxyTrainer | None, args):
        self.cols = [X[:, i] for i in range(X.shape[1])]
        self.N = X.shape[0]
        self.members = [np.where(clusters == c)[0] for c in range(N_CLUSTERS)]
        self.members = [m for m in self.members if len(m)]
        self.pset, self.trainer, self.args = pset, trainer, args
        self.k_total = min(args.k_subset, self.N)
        self.cache: dict[str, float] = load_json(CACHE_PATH, {})
        self.history: list[dict] = []
        self.hits = self.misses = 0
        print(f"[önbellek] diskten {len(self.cache)} kayıt yüklendi")

    def scores(self, ind) -> np.ndarray:
        func = gp.compile(ind, self.pset)
        with np.errstate(all="ignore"):
            s = np.asarray(func(*self.cols), dtype=np.float64)
        s = np.broadcast_to(s, (self.N,)).copy()
        return np.clip(np.nan_to_num(s, nan=0.0, posinf=1e12, neginf=-1e12), -1e12, 1e12)

    def select(self, scores: np.ndarray) -> np.ndarray:
        return stratified_topk(scores, self.members, self.args.per_cluster, self.k_total)

    def _surrogate(self, sel: np.ndarray) -> float:
        """--dry-run: GPU'suz mantık testi için deterministik sahte kayıp."""
        g = np.random.RandomState(int(sel.sum()) % (2**31))
        return 2.0 + 0.1 * g.rand()

    def __call__(self, ind) -> tuple[float]:
        t0 = time.time()
        sel = self.select(self.scores(ind))
        key = subset_key(sel)
        hit = key in self.cache
        if hit:
            self.hits += 1
            val_loss = self.cache[key]
        else:
            self.misses += 1
            try:
                val_loss = self._surrogate(sel) if self.trainer is None else self.trainer.train_and_eval(sel)
                if not np.isfinite(val_loss):
                    val_loss = PENALTY
            except torch.cuda.OutOfMemoryError:
                print("[uyarı] OOM -> bireye ceza verildi")
                free_vram()
                val_loss = PENALTY
            self.cache[key] = float(val_loss)
            save_json(self.cache, CACHE_PATH)
            free_vram()
        fitness = val_loss + BLOAT_COEF * len(ind)
        self.history.append(
            {"expr": str(ind), "infix": to_infix(ind), "size": len(ind), "val_loss": val_loss,
             "fitness": fitness, "cache_hit": hit, "seconds": round(time.time() - t0, 1)}
        )
        print(f"  [birey] boyut={len(ind):3d}  L_val={val_loss:.4f}  fitness={fitness:.4f}  "
              f"{'ÖNBELLEK' if hit else f'{time.time() - t0:.0f} sn'}")
        return (fitness,)


# --------------------------------------------------------------------------- #
# Evrim
# --------------------------------------------------------------------------- #
def setup_toolbox(pset, evaluator: FitnessEvaluator, tournsize: int = 3) -> base.Toolbox:
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", evaluator)
    tb.register("select", tools.selTournament, tournsize=tournsize)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=8)  # üstel ağaç büyümesine karşı
    tb.decorate("mate", limit)
    tb.decorate("mutate", limit)
    return tb


def evolve(tb, pop_size: int, ngen: int, cxpb: float, mutpb: float):
    pop = tb.population(n=pop_size)
    hof = tools.HallOfFame(2)
    stats = tools.Statistics(lambda i: i.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("med", np.median)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "min", "avg", "med"]

    print(f"\n=== Nesil 0 (başlangıç popülasyonu: {pop_size}) ===")
    for ind, fit in zip(pop, map(tb.evaluate, pop)):
        ind.fitness.values = fit
    hof.update(pop)
    logbook.record(gen=0, nevals=len(pop), **stats.compile(pop))
    print(logbook.stream)

    for gen in range(1, ngen + 1):
        print(f"\n=== Nesil {gen}/{ngen} ===")
        offspring = algorithms.varAnd(tb.select(pop, len(pop)), tb, cxpb, mutpb)
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind, fit in zip(invalid, map(tb.evaluate, invalid)):
            ind.fitness.values = fit
        # elitizm: HallOfFame(2) bireyleri sonraki nesle taşınır (en kötü iki birey yerine)
        offspring.sort(key=lambda i: i.fitness.values[0])
        elites = [copy.deepcopy(e) for e in hof]
        offspring[-len(elites):] = elites
        pop[:] = offspring
        hof.update(pop)
        logbook.record(gen=gen, nevals=len(invalid), **stats.compile(pop))
        print(logbook.stream)
    return pop, hof, logbook


def save_subsets(best, evaluator: FitnessEvaluator, pool: pd.DataFrame) -> dict[str, np.ndarray]:
    S = evaluator.scores(best)
    k = evaluator.k_total
    rng = np.random.RandomState(SEED)
    subsets = {
        "gp_strat1k": evaluator.select(S),                                   # önerilen: f_GP* + tabakalı
        "gp_top1k": np.sort(np.argsort(-S, kind="stable")[:k]),              # ablasyon: kümelemesiz
        "ifd_top1k": np.sort(np.argsort(-pool["IFD"].to_numpy(), kind="stable")[:k]),  # baseline
        "rand1k": np.sort(rng.choice(len(pool), size=k, replace=False)),
    }
    for name, idx in subsets.items():
        save_json([int(i) for i in idx], SUBSET_DIR / f"{name}.json")
    np.save(GP_DIR / "gp_scores.npy", S.astype(np.float32))
    names = list(subsets)
    print("\n[alt küme örtüşmesi (Jaccard)]")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            A, B = set(subsets[a].tolist()), set(subsets[b].tolist())
            print(f"  {a:>11s} ~ {b:<11s}: {len(A & B) / max(1, len(A | B)):.3f}")
    return subsets


def plot_evolution(logbook) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = logbook.select("gen")
    plt.figure(figsize=(6, 4))
    plt.plot(gens, logbook.select("min"), "o-", label="en iyi")
    plt.plot(gens, logbook.select("med"), "s--", label="medyan")
    plt.xlabel("Nesil")
    plt.ylabel("Uygunluk = L_val + 0.001·|T|")
    plt.title("GP evrimi (vekil model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GP_DIR / "evolution.png", dpi=150)
    plt.close()


def get_args(argv: list[str] | None = None, **overrides):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--cxpb", type=float, default=0.7)
    ap.add_argument("--mutpb", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=60, help="mikro-SFT gradyan adımı")
    ap.add_argument("--batch", type=int, default=16, help="adım başına etkin toplu boyut")
    ap.add_argument("--micro-batch", type=int, default=8, help="bellek için mikro-toplu (batch bölünür)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k-subset", type=int, default=K_SUBSET)
    ap.add_argument("--per-cluster", type=int, default=PER_CLUSTER)
    ap.add_argument("--dry-run", action="store_true", help="GPU eğitimi yerine sahte kayıp (yalnızca mantık testi)")
    ap.add_argument("--fresh", action="store_true", help="fitness_cache'i yok sayıp sıfırdan başla")
    return parse_overrides(ap, {k.replace("-", "_"): v for k, v in overrides.items()}, argv)


def run_gp_evolution(args=None, **overrides) -> dict:
    """
    Aşama 2'nin tamamı. Dönüş: {"best", "hof", "logbook", "history", "subsets", "evaluator", "pset", "minutes"}.
    Çıktılar GP_DIR altına yazılır.
    """
    args = args or get_args(**overrides)
    set_seed()
    if args.fresh and CACHE_PATH.exists():
        CACHE_PATH.unlink()

    pool = pd.read_parquet(FEAT_DIR / "pool.parquet")
    scaled = pd.read_parquet(FEAT_DIR / "features_scaled.parquet")
    X = scaled[FEATURE_NAMES].to_numpy(dtype=np.float64)
    clusters = scaled["cluster"].to_numpy()
    print(f"[veri] N={len(pool)}, küme sayısı={len(np.unique(clusters))}")

    pset = build_pset()
    trainer = None if args.dry_run else ProxyTrainer(pool, args)
    evaluator = FitnessEvaluator(X, clusters, pset, trainer, args)
    tb = setup_toolbox(pset, evaluator)

    t0 = time.time()
    pop, hof, logbook = evolve(tb, args.pop, args.gens, args.cxpb, args.mutpb)
    best = hof[0]
    minutes = (time.time() - t0) / 60

    print("\n" + "=" * 72)
    print("KEŞFEDİLEN OPTİMAL SEMBOLİK FORMÜL  f_GP*")
    print("  infix  :", to_infix(best))
    print("  prefix :", str(best))
    print(f"  fitness: {best.fitness.values[0]:.5f}   boyut: {len(best)}   süre: {minutes:.1f} dk")
    print(f"  GPU eğitimi: {evaluator.misses} kez, önbellek isabeti: {evaluator.hits}")
    print("=" * 72)

    subsets = save_subsets(best, evaluator, pool)
    history = pd.DataFrame(evaluator.history)
    history.to_csv(GP_DIR / "history.csv", index=False)
    pd.DataFrame(logbook).to_csv(GP_DIR / "logbook.csv", index=False)
    save_json(
        {
            "infix": to_infix(best),
            "prefix": str(best),
            "fitness": best.fitness.values[0],
            "size": len(best),
            "hall_of_fame": [{"infix": to_infix(h), "prefix": str(h), "fitness": h.fitness.values[0]} for h in hof],
            "args": vars(args),
            "minutes": minutes,
            "gpu_trainings": evaluator.misses,
            "cache_hits": evaluator.hits,
            "subset_sizes": {k: len(v) for k, v in subsets.items()},
        },
        GP_DIR / "best_formula.json",
    )
    plot_evolution(logbook)
    print(f"[tamam] çıktılar: {GP_DIR}")
    return {"best": best, "hof": hof, "logbook": logbook, "history": history, "subsets": subsets,
            "evaluator": evaluator, "pset": pset, "minutes": minutes}
