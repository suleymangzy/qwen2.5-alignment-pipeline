"""
Modül 1 - Anlamsal kümeleme ve 6 öznitelikli matrisin çıkarılması (çevrimdışı).

Adım 1: D_havuz içindeki her (x_i, y_i) çifti BAAI/bge-small-en-v1.5 ile 384 boyutlu uzaya gömülür,
        KMeans(n_clusters=20, random_state=42) ile M=20 kümeye ayrılır; küme etiketi c(i) ve
        küme merkezine kosinüs uzaklığı kaydedilir.
Adım 2: Qwen2.5-0.5B-Base üzerinde torch.no_grad() altında, her çift için
        Lx, Ly, PPL, IFD, Hy, Ddiv hesaplanır:
          * Koşullu geçiş  (istem + yanıt, istem simgeleri -100)  -> PPL, CE(y|x), Hy
          * Koşulsuz geçiş (yalnızca yanıt, başa EOS simgesi)     -> CE(y)   (IFD paydası)
        Tüm veri kümesi tek bir ileri besleme turunda (iki ileri geçiş: koşullu + koşulsuz) işlenir;
        IFD'nin tanımı gereği koşulsuz kayıp ayrı bir geçiş gerektirir.
        Tüm öznitelikler MinMaxScaler(0, 1) ile normalize edilip Parquet/NumPy olarak kaydedilir.

Kullanım (defter içinden):
    from features import run_feature_extraction
    pool, scaled = run_feature_extraction(limit=0)
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from common import (
    EMBED_MODEL,
    FEAT_DIR,
    FEATURE_NAMES,
    N_CLUSTERS,
    PROXY_MODEL,
    SEED,
    free_vram,
    load_alpaca_pool,
    load_causal_lm,
    load_tokenizer,
    parse_overrides,
    set_seed,
    tokenize_pairs,
    PadCollator,
)

EMB_PATH = FEAT_DIR / "embeddings.npy"
POOL_PATH = FEAT_DIR / "pool.parquet"
SCALED_PQ = FEAT_DIR / "features_scaled.parquet"
SCALED_NP = FEAT_DIR / "features_scaled.npy"


def get_args(argv: list[str] | None = None, **overrides):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Duman testi için ilk N örnek (0 = tümü)")
    ap.add_argument("--force", action="store_true", help="Var olan çıktıları yeniden üret")
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--token-budget", type=int, default=16384, help="Bir toplu işteki en fazla (B x T) simge")
    ap.add_argument("--max-batch", type=int, default=64)
    return parse_overrides(ap, {k.replace("-", "_"): v for k, v in overrides.items()}, argv)


# --------------------------------------------------------------------------- #
# Adım 1: gömme + KMeans
# --------------------------------------------------------------------------- #
def embed_pool(df: pd.DataFrame, batch_size: int, force: bool) -> np.ndarray:
    if EMB_PATH.exists() and not force:
        emb = np.load(EMB_PATH).astype(np.float32)
        if len(emb) == len(df):
            print(f"[gömme] önbellekten yüklendi: {emb.shape}")
            return emb
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cuda":
        st.half()
    texts = (df["prompt"] + "\n" + df["response"]).tolist()
    emb = st.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    np.save(EMB_PATH, emb)
    del st
    free_vram()
    print(f"[gömme] {emb.shape} (384 boyutlu beklenir)")
    return emb


def cluster_embeddings(emb: np.ndarray):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(emb)
    centroids = km.cluster_centers_.astype(np.float32)
    cnorm = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    # gömmeler birim normlu -> kosinüs uzaklığı = 1 - <e_i, mu_c(i)/|mu_c(i)|>
    cos_dist = 1.0 - np.einsum("nd,nd->n", emb, cnorm[labels])
    np.save(FEAT_DIR / "centroids.npy", centroids)
    return labels, np.clip(cos_dist, 0.0, 2.0)


# --------------------------------------------------------------------------- #
# Adım 2: 6 öznitelik
# --------------------------------------------------------------------------- #
def make_batches(lengths: list[int], budget: int, max_bs: int) -> list[list[int]]:
    order = np.argsort(lengths)[::-1]
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def response_stats(model, batch: dict, chunk: int = 4096):
    """
    Yanıt simgeleri üzerinde simge başı NLL toplamı, entropi toplamı ve simge sayısı.
    lm_head yalnızca yanıt konumlarına uygulanır (150k'lık sözlükte bellek tasarrufu).
    """
    dev = next(model.parameters()).device
    ids, att, lab = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "labels"))
    hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    b_idx, t_idx = mask.nonzero(as_tuple=True)
    nll_sum = torch.zeros(ids.size(0), device=dev, dtype=torch.float32)
    ent_sum = torch.zeros_like(nll_sum)
    for s in range(0, b_idx.numel(), chunk):
        bi, ti = b_idx[s : s + chunk], t_idx[s : s + chunk]
        logp = torch.log_softmax(model.lm_head(hidden[bi, ti]).float(), dim=-1)
        nll = -logp.gather(1, tgt[bi, ti].unsqueeze(1)).squeeze(1)
        ent = -(logp.exp() * logp).sum(-1)
        nll_sum.index_add_(0, bi, nll)
        ent_sum.index_add_(0, bi, ent)
    return nll_sum.cpu().numpy(), ent_sum.cpu().numpy(), mask.sum(1).cpu().numpy()


def run_pass(model, examples: list[dict], collate, budget: int, max_bs: int, desc: str):
    n = len(examples)
    nll, ent, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    lengths = [len(e["input_ids"]) for e in examples]
    for idxs in tqdm(make_batches(lengths, budget, max_bs), desc=desc):
        s_nll, s_ent, s_cnt = response_stats(model, collate([examples[i] for i in idxs]))
        nll[idxs], ent[idxs], cnt[idxs] = s_nll, s_ent, s_cnt
    cnt = np.maximum(cnt, 1)
    return nll / cnt, ent / cnt, cnt


def extract_features(df: pd.DataFrame, cos_dist: np.ndarray, args) -> pd.DataFrame:
    tok = load_tokenizer(PROXY_MODEL)
    from common import format_prompt

    cond = tokenize_pairs(tok, [format_prompt(p) for p in df["prompt"]], df["response"].tolist())
    # koşulsuz örnekler: [EOS] + yanıt, yalnızca yanıt simgeleri etiketli
    uncond = []
    for e in cond:
        resp = e["input_ids"][e["prompt_len"] :]
        uncond.append({"input_ids": [tok.eos_token_id] + resp, "labels": [-100] + resp})
    ttr = np.array(
        [len(set(e["input_ids"][e["prompt_len"] :])) / max(1, len(e["input_ids"]) - e["prompt_len"]) for e in cond]
    )

    model = load_causal_lm(PROXY_MODEL, quantize=False)  # öznitelikler tam duyarlıkta (bf16) hesaplanır
    model.eval()
    collate = PadCollator(tok.pad_token_id)
    nll_c, ent_c, _ = run_pass(model, cond, collate, args.token_budget, args.max_batch, "koşullu geçiş")
    nll_u, _, _ = run_pass(model, uncond, collate, args.token_budget, args.max_batch, "koşulsuz geçiş")
    del model
    free_vram()

    feats = pd.DataFrame(
        {
            "Lx": [e["raw_lx"] for e in cond],
            "Ly": [e["raw_ly"] for e in cond],
            "PPL": np.exp(np.minimum(nll_c, 50.0)),
            "IFD": nll_c / np.maximum(nll_u, 1e-6),
            "Hy": ent_c,
            "Ddiv": ttr * cos_dist,
        },
        index=df.index,
    )
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats.fillna(feats.median())


def run_feature_extraction(args=None, **overrides):
    """Aşama 1'in tamamı. (pool_df, scaled_df) döndürür; çıktılar FEAT_DIR altına yazılır."""
    args = args or get_args(**overrides)
    set_seed()
    if POOL_PATH.exists() and SCALED_PQ.exists() and not args.force:
        print("[bilgi] Çıktılar zaten var, diskten yükleniyor (yeniden üretmek için force=True).")
        return pd.read_parquet(POOL_PATH), pd.read_parquet(SCALED_PQ)

    df = load_alpaca_pool()
    if args.limit:
        df = df.iloc[: args.limit].copy()
        print(f"[duman testi] yalnızca ilk {len(df)} örnek işleniyor")
    print(f"[veri] D_havuz = {len(df)} örnek")

    emb = embed_pool(df, args.embed_batch, args.force or bool(args.limit))
    labels, cos_dist = cluster_embeddings(emb)
    print("[kümeleme] küme boyutları:", np.bincount(labels, minlength=N_CLUSTERS).tolist())

    feats = extract_features(df, cos_dist, args)

    pool = pd.concat([df, feats], axis=1)
    pool.insert(2, "cluster", labels.astype(np.int16))
    pool.insert(3, "cos_dist", cos_dist.astype(np.float32))
    pool.to_parquet(POOL_PATH)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(feats[FEATURE_NAMES].to_numpy(dtype=np.float64))
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_NAMES, index=df.index)
    scaled_df["cluster"] = labels.astype(np.int16)
    scaled_df.to_parquet(SCALED_PQ)
    np.save(SCALED_NP, scaled.astype(np.float32))
    joblib.dump(scaler, FEAT_DIR / "minmax_scaler.joblib")

    summary = pool.groupby("cluster")[FEATURE_NAMES].mean().assign(size=pool.groupby("cluster").size())
    summary.to_csv(FEAT_DIR / "cluster_summary.csv")
    print(summary.round(3).to_string())
    print(f"[tamam] kaydedildi: {POOL_PATH.name}, {SCALED_PQ.name}, {SCALED_NP.name}")
    return pool, scaled_df
