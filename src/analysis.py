"""
Akademik raporlama yardımcıları (defterler tarafından kullanılır).

  * Tekrarlanabilirlik: environment_report()
  * Şekil/tablo standardı: set_academic_style(), show_figure(), show_table()
        - Şekiller 300 dpi PNG + vektörel PDF olarak outputs/figures altına, tablolar CSV + LaTeX olarak
          outputs/tables altına yazılır; "Şekil N" / "Tablo N" numaraları ve açıklamaları otomatik verilir.
  * Aşama 1-4'e özgü tablo ve çizim fonksiyonları (öznitelik istatistikleri, küme görselleştirmesi, GP yakınsaması,
    alt küme profilleri, eğitim eğrileri, maliyet, kazanma oranı güven aralıkları).
Tüm fonksiyonlar yalnızca diskteki çıktıları okur; GPU gerektirmez.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ALIGNED_MODELS,
    ALL_MODELS,
    FEAT_DIR,
    FEATURE_NAMES,
    FIG_DIR,
    GP_DIR,
    METRIC_DIR,
    N_CLUSTERS,
    OUT,
    SEED,
    SFT_MODELS,
    load_json,
    save_json,
)

TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT / "report_index.json"

FEATURE_LABELS = {
    "Lx": r"$L_x$ (istem simgesi)", "Ly": r"$L_y$ (yanıt simgesi)", "PPL": "PPL",
    "IFD": "IFD", "Hy": r"$H_y$ (nat)", "Ddiv": r"$D_{div}$",
}
PRETTY = {"Model_SFT_Full": "SFT-Full", "Model_SFT_Rand1k": "SFT-Rand1k", "Model_SFT_IFD1k": "SFT-IFD1k",
          "Model_SFT_GP_Top1k": "SFT-GP-Top1k", "Model_SFT_GP_Strat1k": "SFT-GP-Strat1k"}
PRETTY.update({m: m.replace("Model_", "").replace("_", "-") for m in ALIGNED_MODELS})


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("Model_", ""))


# --------------------------------------------------------------------------- #
# Tekrarlanabilirlik
# --------------------------------------------------------------------------- #
def environment_report() -> pd.DataFrame:
    """Sürüm, donanım ve rastgelelik tohumu bilgisini içeren tekrarlanabilirlik tablosu."""
    import importlib.metadata as md

    import torch

    rows = [("Python", platform.python_version()), ("İşletim sistemi", platform.platform()), ("Rastgele tohum", SEED)]
    for pkg in ["torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "sentence-transformers",
                "scikit-learn", "deap", "numpy", "pandas", "accelerate", "openai"]:
        try:
            rows.append((pkg, md.version(pkg)))
        except md.PackageNotFoundError:
            rows.append((pkg, "kurulu değil"))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rows += [("GPU", p.name), ("GPU belleği (GB)", round(p.total_memory / 1024**3, 1)), ("CUDA", torch.version.cuda)]
    else:
        rows.append(("GPU", "yok (CPU)"))
    return pd.DataFrame(rows, columns=["bileşen", "sürüm / değer"])


# --------------------------------------------------------------------------- #
# Biçim, numaralandırma, kaydetme
# --------------------------------------------------------------------------- #
def set_academic_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                                              "#F0E442", "#000000"]),
        "mathtext.fontset": "dejavuserif",
    })


def _register(kind: str, name: str, caption: str) -> int:
    idx = load_json(INDEX_PATH, {"figure": {}, "table": {}})
    bucket = idx[kind]
    if name not in bucket:
        bucket[name] = {"no": len(bucket) + 1, "caption": caption}
    else:
        bucket[name]["caption"] = caption
    save_json(idx, INDEX_PATH)
    return bucket[name]["no"]


def show_figure(fig, name: str, caption: str) -> None:
    """Şekli PNG(300 dpi)+PDF kaydeder, defterde gösterir ve numaralı açıklamasını basar."""
    from IPython.display import Markdown, display
    import matplotlib.pyplot as plt

    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    no = _register("figure", name, caption)
    plt.show()
    display(Markdown(f"**Şekil {no}.** {caption}"))
    plt.close(fig)


def show_table(df: pd.DataFrame, name: str, caption: str, index: bool = True, floatfmt: str = "{:.3f}") -> None:
    """Tabloyu CSV + LaTeX olarak kaydeder ve numaralı açıklamasıyla gösterir."""
    from IPython.display import Markdown, display

    df.to_csv(TABLE_DIR / f"{name}.csv", index=index, encoding="utf-8-sig")
    try:
        latex = df.to_latex(index=index, float_format=lambda v: floatfmt.format(v), caption=caption, label=f"tab:{name}",
                            escape=True)
        (TABLE_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")
    except Exception as e:  # LaTeX çıktısı yardımcıdır; hata raporu durdurmamalı
        print(f"[uyarı] LaTeX yazılamadı: {e}")
    no = _register("table", name, caption)
    display(Markdown(f"**Tablo {no}.** {caption}"))
    display(df.style.format(floatfmt, na_rep="–"))


# --------------------------------------------------------------------------- #
# Aşama 1: öznitelikler ve kümeler
# --------------------------------------------------------------------------- #
def feature_summary(pool: pd.DataFrame) -> pd.DataFrame:
    """Ham (ölçeklenmemiş) 6 özniteliğin betimsel istatistikleri."""
    d = pool[FEATURE_NAMES].describe(percentiles=[0.05, 0.5, 0.95]).T
    d["skew"] = pool[FEATURE_NAMES].skew()
    d.index = [FEATURE_LABELS[i] for i in d.index]
    return d[["count", "mean", "std", "min", "5%", "50%", "95%", "max", "skew"]]


def feature_sanity(pool: pd.DataFrame, scaled: pd.DataFrame) -> pd.DataFrame:
    """Beklenen matematiksel özelliklerin otomatik doğrulaması (rapor için 'geçti/kaldı' tablosu)."""
    checks = {
        "Tüm öznitelikler sonlu": bool(np.isfinite(pool[FEATURE_NAMES].to_numpy()).all()),
        "PPL ≥ 1": bool((pool["PPL"] >= 1 - 1e-6).all()),
        "IFD > 0": bool((pool["IFD"] > 0).all()),
        "Hy ≥ 0": bool((pool["Hy"] >= 0).all()),
        "Ddiv ≥ 0": bool((pool["Ddiv"] >= 0).all()),
        "Ölçekli değerler [0,1] içinde": bool(((scaled[FEATURE_NAMES] >= -1e-9) & (scaled[FEATURE_NAMES] <= 1 + 1e-9)).all().all()),
        f"Küme sayısı = {N_CLUSTERS}": pool["cluster"].nunique() == N_CLUSTERS,
        "Boş küme yok": bool((pool["cluster"].value_counts() > 0).all()),
    }
    return pd.DataFrame({"kontrol": list(checks), "sonuç": ["geçti" if v else "KALDI" for v in checks.values()]}).set_index("kontrol")


def spearman_corr(pool: pd.DataFrame) -> pd.DataFrame:
    c = pool[FEATURE_NAMES].corr(method="spearman")
    c.index = c.columns = FEATURE_NAMES
    return c


def plot_feature_distributions(pool: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, f in zip(axes.ravel(), FEATURE_NAMES):
        x = pool[f].to_numpy()
        lo, hi = np.percentile(x, [0.5, 99.5])
        ax.hist(np.clip(x, lo, hi), bins=60, color="#0072B2", alpha=0.85)
        ax.axvline(np.median(x), color="#D55E00", ls="--", lw=1, label=f"medyan={np.median(x):.2f}")
        ax.set_xlabel(FEATURE_LABELS[f])
        ax.set_ylabel("örnek sayısı")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"Spearman $\rho$")
    fig.tight_layout()
    return fig


def plot_cluster_projection(labels: np.ndarray, emb: np.ndarray | None = None, max_points: int = 8000):
    """384 boyutlu gömmelerin PCA ile 2B izdüşümü (küme rengiyle)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    emb = np.load(FEAT_DIR / "embeddings.npy").astype(np.float32) if emb is None else emb
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(emb), size=min(max_points, len(emb)), replace=False)
    pca = PCA(n_components=2, random_state=SEED).fit(emb)
    z = pca.transform(emb[idx])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=labels[idx], cmap="tab20", s=4, alpha=0.7, linewidths=0)
    ax.set_xlabel(f"PC1 (%{100 * pca.explained_variance_ratio_[0]:.1f})")
    ax.set_ylabel(f"PC2 (%{100 * pca.explained_variance_ratio_[1]:.1f})")
    fig.colorbar(sc, ax=ax, label="küme", ticks=range(0, N_CLUSTERS, 2))
    fig.tight_layout()
    return fig


def cluster_table(pool: pd.DataFrame, exemplar_chars: int = 70) -> pd.DataFrame:
    """Küme boyutu, ortalama öznitelikler ve merkeze en yakın örnek istem (nitel yorum için)."""
    g = pool.groupby("cluster")
    t = g[FEATURE_NAMES].mean()
    t.insert(0, "n", g.size())
    t["merkeze_en_yakın_istem"] = [
        pool.loc[pool[pool["cluster"] == c]["cos_dist"].idxmin(), "prompt"].replace("\n", " ")[:exemplar_chars]
        for c in t.index
    ]
    return t


# --------------------------------------------------------------------------- #
# Aşama 2: GP
# --------------------------------------------------------------------------- #
def formula_latex(best, simplify: bool = True) -> str:
    """DEAP ağacını (korumalı operatörlerle) SymPy'a çevirip LaTeX üretir."""
    import sympy as sp
    from deap import gp

    syms = {n: sp.Symbol(n) for n in FEATURE_NAMES}

    def rec(i):
        node = best[i]
        if isinstance(node, gp.Terminal):
            v = node.value
            return (syms[v] if isinstance(v, str) else sp.Float(round(float(v), 4))), i + 1
        args, j = [], i + 1
        for _ in range(node.arity):
            a, j = rec(j)
            args.append(a)
        ops = {
            "add": lambda a: a[0] + a[1], "sub": lambda a: a[0] - a[1], "mul": lambda a: a[0] * a[1],
            "pdiv": lambda a: a[0] / a[1], "plog": lambda a: sp.log(sp.Abs(a[0]) + sp.Float(1e-6)),
            "psqrt": lambda a: sp.sqrt(sp.Abs(a[0])), "max": lambda a: sp.Max(*a), "min": lambda a: sp.Min(*a),
        }
        return ops[node.name](args), j

    expr = rec(0)[0]
    if simplify:
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
    return sp.latex(expr)


def plot_gp_convergence(logbook):
    import matplotlib.pyplot as plt

    get = (lambda k: logbook.select(k)) if hasattr(logbook, "select") else (lambda k: list(logbook[k]))
    gens = get("gen")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(gens, get("min"), "o-", label="en iyi birey")
    ax.plot(gens, get("med"), "s--", label="popülasyon medyanı")
    ax.plot(gens, get("avg"), "^:", label="popülasyon ortalaması")
    ax.set_xlabel("nesil")
    ax.set_ylabel(r"uygunluk $=\mathcal{L}_{val}+0.001\,|T|$")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fitness_landscape(history: pd.DataFrame):
    """Değerlendirilen her bireyin L_val değeri ve ağaç boyutu (bloat kontrolü etkisi)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    h = history.reset_index(drop=True)
    ax = axes[0]
    ax.scatter(h.index[~h["cache_hit"]], h["val_loss"][~h["cache_hit"]], s=14, label="GPU mikro-SFT", color="#0072B2")
    ax.scatter(h.index[h["cache_hit"]], h["val_loss"][h["cache_hit"]], s=14, label="önbellek isabeti", color="#D55E00", marker="x")
    ax.set_xlabel("değerlendirme sırası")
    ax.set_ylabel(r"$\mathcal{L}_{val}$")
    ax.legend()
    sc = axes[1].scatter(h["size"], h["val_loss"], c=h.index, cmap="viridis", s=14)
    axes[1].set_xlabel(r"ağaç boyutu $|T_j|$ (düğüm)")
    axes[1].set_ylabel(r"$\mathcal{L}_{val}$")
    fig.colorbar(sc, ax=axes[1], label="değerlendirme sırası")
    fig.tight_layout()
    return fig


def subset_profile_table(pool: pd.DataFrame, subsets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Her alt kümenin ham öznitelik ortalamaları, kümeler arası dağılım entropisi ve kapsama."""
    rows = {}
    for name, idx in subsets.items():
        sub = pool.iloc[np.asarray(idx)]
        counts = sub["cluster"].value_counts(normalize=True).reindex(range(N_CLUSTERS), fill_value=0).to_numpy()
        ent = -(counts[counts > 0] * np.log(counts[counts > 0])).sum() / np.log(N_CLUSTERS)
        rows[name] = {**{f: sub[f].mean() for f in FEATURE_NAMES}, "n": len(sub),
                      "kapsanan_küme": int((counts > 0).sum()), "küme_entropisi (0-1)": ent}
    return pd.DataFrame(rows).T


def jaccard_table(subsets: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(subsets)
    sets = {n: set(map(int, subsets[n])) for n in names}
    return pd.DataFrame([[len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for b in names] for a in names],
                        index=names, columns=names)


def plot_subset_clusters(pool: pd.DataFrame, subsets: dict[str, np.ndarray]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(subsets)
    for k, (name, idx) in enumerate(subsets.items()):
        c = pool.iloc[np.asarray(idx)]["cluster"].value_counts().reindex(range(N_CLUSTERS), fill_value=0)
        ax.bar(np.arange(N_CLUSTERS) + k * w, c.to_numpy(), width=w, label=name)
    ax.set_xticks(np.arange(N_CLUSTERS) + 0.4 - w / 2, range(N_CLUSTERS))
    ax.set_xlabel("anlamsal küme")
    ax.set_ylabel("seçilen örnek sayısı")
    ax.legend(ncol=4)
    fig.tight_layout()
    return fig


def plot_score_vs_ifd(pool: pd.DataFrame, scores: np.ndarray, subsets: dict[str, np.ndarray]):
    """f_GP* skorunun IFD ile ilişkisi; seçilen alt kümeler vurgulanır."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    rho = spearmanr(scores, pool["IFD"]).statistic
    rng = np.random.RandomState(SEED)
    bg = rng.choice(len(pool), size=min(6000, len(pool)), replace=False)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(pool["IFD"].iloc[bg], scores[bg], s=3, color="lightgray", label="havuz")
    for name, color in (("gp_strat1k", "#D55E00"), ("gp_top1k", "#0072B2")):
        if name in subsets:
            idx = np.asarray(subsets[name])
            ax.scatter(pool["IFD"].iloc[idx], scores[idx], s=5, color=color, label=name)
    ax.set_xlabel("IFD (ham)")
    ax.set_ylabel(r"$S_i = f_{GP}^*(\cdot)$")
    ax.set_title(rf"Spearman $\rho$(IFD, $S$) = {rho:.2f}")
    ax.legend(markerscale=3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Aşama 3: eğitim eğrileri ve maliyet
# --------------------------------------------------------------------------- #
def load_logs() -> dict[str, list[dict]]:
    logs = {n: load_json(METRIC_DIR / f"{n}_log.json", []) for n in ALL_MODELS}
    return {n: l for n, l in logs.items() if l}


def _smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def plot_curves(logs: dict, models: list[str], key: str, ylabel: str, title: str | None = None, smooth: int = 5):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    plotted = False
    for m in models:
        pts = [(r["step"], r[key]) for r in logs.get(m, []) if key in r and "step" in r and np.isfinite(r[key])]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, _smooth(y, smooth), label=pretty(m))
            plotted = True
    ax.set_xlabel("optimizasyon adımı")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if plotted:
        ax.legend()
    fig.tight_layout()
    return fig


def cost_table() -> pd.DataFrame:
    costs = load_json(METRIC_DIR / "training_costs.json", {})
    rows = []
    for n in ALL_MODELS:
        stage = "sft" if n in SFT_MODELS else ALIGNED_MODELS[n][1]
        c = costs.get(f"{stage}:{n}")
        if c:
            rows.append({"model": pretty(n), "aşama": stage.upper(), "süre (dk)": c["seconds"] / 60,
                         "tepe VRAM (GB)": c["peak_vram_gb"]})
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame(columns=["aşama", "süre (dk)", "tepe VRAM (GB)"])


def plot_cost(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(df.index, df["tepe VRAM (GB)"], color="#009E73")
    axes[0].set_ylabel("tepe VRAM (GB)")
    axes[1].bar(df.index, df["süre (dk)"], color="#E69F00")
    axes[1].set_ylabel("süre (dk, log ölçek)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Aşama 4: değerlendirme
# --------------------------------------------------------------------------- #
def eval_table() -> pd.DataFrame:
    m = load_json(METRIC_DIR / "eval_metrics.json", {})
    rows = [{"model": pretty(n), "ort. uzunluk (simge)": m[n]["mean_len_tokens"], "medyan uzunluk": m[n]["median_len_tokens"],
             "max uzunluğa ulaşan %": 100 * m[n]["frac_hit_max_len"], "D_KL (nat/simge)": m[n]["kl_to_ref"],
             "KL referansı": "taban" if m[n]["kl_ref"] == "base" else pretty(m[n]["kl_ref"])}
            for n in ALL_MODELS if n in m]
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def plot_length_kl(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(df.index, df["ort. uzunluk (simge)"], color="#0072B2")
    axes[0].set_ylabel("ortalama yanıt uzunluğu (simge)")
    axes[1].bar(df.index, df["D_KL (nat/simge)"], color="#D55E00")
    axes[1].set_ylabel(r"$D_{KL}(\pi\,\|\,\pi_{ref})$ (nat/simge)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    return fig


def pairwise_table(pw: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({
        "model": pw["model"].map(pretty), "rakip": pw["opponent"].map(pretty),
        "ham kazanma %": pw["raw_winrate"], "ham %95 GA": [f"[{a:.1f}, {b:.1f}]" for a, b in zip(pw["raw_lo"], pw["raw_hi"])],
        "LC kazanma %": pw["lc_winrate"], "LC %95 GA": [f"[{a:.1f}, {b:.1f}]" for a, b in zip(pw["lc_lo"], pw["lc_hi"])],
        "K/B/Y": [f"{w}/{t}/{l}" for w, t, l in zip(pw["wins"], pw["ties"], pw["losses"])],
        "işaret testi p": pw["sign_test_p"], "sıra tutarlılığı": pw["order_consistency"],
    })
    return t.set_index(["model", "rakip"])


def plot_forest(pw: pd.DataFrame):
    """LC kazanma oranları ve %95 bootstrap güven aralıkları (50 = eşitlik çizgisi)."""
    import matplotlib.pyplot as plt

    labels = [f"{pretty(a)} vs {pretty(b)}" for a, b in zip(pw["model"], pw["opponent"])]
    y = np.arange(len(pw))[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(pw) + 1.6))
    ax.errorbar(pw["lc_winrate"], y, xerr=[pw["lc_winrate"] - pw["lc_lo"], pw["lc_hi"] - pw["lc_winrate"]],
                fmt="o", color="#0072B2", capsize=3, label="LC kazanma %")
    ax.scatter(pw["raw_winrate"], y, marker="x", color="#D55E00", label="ham kazanma %")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("kazanma oranı (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_lc_heatmap(mat: pd.DataFrame):
    import matplotlib.pyplot as plt

    labels = [pretty(m) for m in mat.index]
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 3, 0.8 * len(labels) + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, label="LC kazanma % (satır > sütun)")
    fig.tight_layout()
    return fig


def position_bias_table(bias: dict) -> pd.DataFrame:
    tot = max(1, sum(bias.values()))
    return pd.DataFrame({"yargı sayısı": bias, "oran (%)": {k: 100 * v / tot for k, v in bias.items()}}).rename(
        index={"A": "A konumu seçildi", "B": "B konumu seçildi", "tie": "beraberlik"})
