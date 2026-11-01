"""
Modül 0 - Ortak yapılandırma ve yardımcı fonksiyonlar.

Tüm aşamalar (01..04) bu modülü içe aktarır. İçerik:
  * Sabitler (model adları, dosya yolları, küme sayısı, alt küme boyutları)
  * Bellek yönetimi (free_vram, track_cost) ve 4-bit NF4 nicemleme yapılandırması
  * Veri kümesi yükleyicileri (alpaca-cleaned, no_robots tabakalı 250/250 bölünmesi,
    ultrafeedback_binarized)
  * İstem biçimlendirme, simgeleştirme (prompt maskeli) ve toplu birleştirici (collator)
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------- #
# Yollar
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
OUT = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
FEAT_DIR = OUT / "features"
GP_DIR = OUT / "gp"
MODEL_DIR = OUT / "models"
GEN_DIR = OUT / "generations"
METRIC_DIR = OUT / "metrics"
FIG_DIR = OUT / "figures"
JUDGE_DIR = OUT / "judge"
for _d in (FEAT_DIR, GP_DIR, MODEL_DIR, GEN_DIR, METRIC_DIR, FIG_DIR, JUDGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def export_outputs(name: str = "outputs_backup", include_models: bool = True, download: bool = True) -> Path | None:
    """Colab VM diskindeki OUT klasörünü tek bir zip'e paketler ve yerel makineye indirmeyi dener.

    Colab uzantısında çekirdek uzak makinede çalışır ve VM kapanınca outputs/ silinir; VM yerel diske yazamaz.
    Bu nedenle her defterin sonunda çağrılır. İndirme otomatik başlamazsa zip, VS Code'un Colab
    "Contents" görünümünden (/content/<name>.zip) indirilebilir. Yerelde çalışırken çıktılar zaten
    yerel olduğundan hiçbir şey yapmaz. Yerelde geri yüklemek için zip'i proje köküne açın.
    """
    import sys
    import zipfile

    if "google.colab" not in sys.modules:
        print("Yerel çalışma: çıktılar zaten", OUT)
        return None

    zip_path = Path("/content") / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(part.startswith("checkpoint-") for part in rel.parts):
                continue  # ara kontrol noktaları (optimizer durumu) büyük ve yeniden üretilebilir
            if not include_models and MODEL_DIR in p.parents:
                continue
            zf.write(p, rel.as_posix())
    print(f"Paketlendi: {zip_path} ({zip_path.stat().st_size / 2**20:,.1f} MB)")
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[uyarı] Otomatik indirme başarısız ({type(exc).__name__}); zip'i Colab 'Contents' görünümünden indirin.")
    return zip_path

# --------------------------------------------------------------------------- #
# Sabitler
# --------------------------------------------------------------------------- #
SEED = 42
PROXY_MODEL = os.environ.get("PROXY_MODEL", "Qwen/Qwen2.5-0.5B")  # Qwen2.5-0.5B-Base
TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-3B")  # Qwen2.5-3B-Base
REWARD_MODEL = os.environ.get("REWARD_MODEL", "Ray2333/GRM-Llama3-8B-rewardmodel-ft")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
JUDGE_MODEL = "gpt-4o"

ALPACA_ID = "yahma/alpaca-cleaned"
NO_ROBOTS_ID = "HuggingFaceH4/no_robots"
ULTRAFEEDBACK_ID = "HuggingFaceH4/ultrafeedback_binarized"

N_CLUSTERS = 20
PER_CLUSTER = 50
K_SUBSET = N_CLUSTERS * PER_CLUSTER  # 1000
N_VAL = 250
N_PREF = 10_000

FEATURE_NAMES = ["Lx", "Ly", "PPL", "IFD", "Hy", "Ddiv"]

SFT_MODELS = [
    "Model_SFT_Full",
    "Model_SFT_Rand1k",
    "Model_SFT_IFD1k",
    "Model_SFT_GP_Top1k",
    "Model_SFT_GP_Strat1k",
]
# Hizalanmış model adı -> (kaynak SFT modeli, yöntem)
ALIGNED_MODELS = {
    "Model_IFD_GRPO": ("Model_SFT_IFD1k", "grpo"),
    "Model_GP_GRPO": ("Model_SFT_GP_Strat1k", "grpo"),
    "Model_IFD_DPO": ("Model_SFT_IFD1k", "dpo"),
    "Model_GP_DPO": ("Model_SFT_GP_Strat1k", "dpo"),
    "Model_IFD_RDPO": ("Model_SFT_IFD1k", "rdpo"),
    "Model_GP_RDPO": ("Model_SFT_GP_Strat1k", "rdpo"),
}
ALL_MODELS = SFT_MODELS + list(ALIGNED_MODELS)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 512
MAX_PROMPT_LEN = 256


# --------------------------------------------------------------------------- #
# Genel yardımcılar
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def free_vram() -> None:
    """Python çöp toplayıcısını ve CUDA önbelleğini temizler."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def has_cuda() -> bool:
    return torch.cuda.is_available()


def compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def bnb_config():
    """4-bit NF4 + çift nicemleme. GPU yoksa (yerel duman testi) None döner."""
    if not torch.cuda.is_available():
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_tokenizer(name: str, padding_side: str = "right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side=padding_side)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    return {key: dtype}


def load_causal_lm(name: str, quantize: bool = True, dtype: torch.dtype | None = None):
    """Nedensel dil modelini (varsayılan: 4-bit NF4) tek GPU'ya yükler."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = _dtype_kwarg(dtype or compute_dtype())
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}
        if quantize:
            kwargs["quantization_config"] = bnb_config()
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.config.use_cache = False
    return model


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )


@contextmanager
def track_cost(name: str, stage: str):
    """Bir eğitim/üretim bloğunun süre ve en yüksek VRAM maliyetini kaydeder."""
    free_vram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rec: dict[str, Any] = {"name": name, "stage": stage}
    t0 = time.time()
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        rec["peak_vram_gb"] = (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0.0
        )
        path = METRIC_DIR / "training_costs.json"
        costs = load_json(path, {})
        costs[f"{stage}:{name}"] = rec
        save_json(costs, path)
        print(f"[maliyet] {stage}:{name} -> {rec['seconds']} sn, tepe VRAM {rec['peak_vram_gb']} GB")


# --------------------------------------------------------------------------- #
# İstem biçimleri
# --------------------------------------------------------------------------- #
def format_prompt(x: str) -> str:
    """Taban (base) model için sade Alpaca tarzı istem şablonu."""
    return f"### Instruction:\n{x.strip()}\n\n### Response:\n"


def merge_instruction_input(instruction: str, inp: str | None) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    return f"{instruction}\n\n{inp}" if inp else instruction


def _as_messages(messages) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in list(messages)]


def format_dialog_prompt(messages) -> str:
    """Çok turlu (system/user/assistant) bir konuşmanın son kullanıcı turuna kadar şablonu."""
    parts = []
    for m in _as_messages(messages):
        if m["role"] == "system":
            parts.append(f"### System:\n{m['content'].strip()}\n\n")
        elif m["role"] == "user":
            parts.append(f"### Instruction:\n{m['content'].strip()}\n\n### Response:\n")
        else:
            parts.append(f"{m['content'].strip()}\n\n")
    return "".join(parts)


def dialog_plain(messages) -> str:
    """Hakem (judge) için okunabilir düz metin konuşma bağlamı."""
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"[{names[m['role']]}]\n{m['content'].strip()}" for m in _as_messages(messages))


# --------------------------------------------------------------------------- #
# Veri kümeleri
# --------------------------------------------------------------------------- #
def load_alpaca_pool() -> pd.DataFrame:
    """D_havuz: istem x_i = instruction (+ input), yanıt y_i = output."""
    from datasets import load_dataset

    ds = load_dataset(ALPACA_ID, split="train")
    df = ds.to_pandas()
    df["prompt"] = [merge_instruction_input(a, b) for a, b in zip(df["instruction"], df["input"])]
    df["response"] = df["output"].astype(str).str.strip()
    df = df[["prompt", "response"]].reset_index(drop=True)
    df.index.name = "idx"
    return df


def load_no_robots_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    no_robots test bölümünden `category` üzerinden tabakalı örnekleme:
      * D_val-250  -> GP uygunluk (fitness) doğrulaması
      * D_test-250 -> kalan örnekler; nihai nicel/nitel değerlendirme (sızıntı önlemek için ayrı)
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    ds = load_dataset(NO_ROBOTS_ID, split="test")
    rows = []
    for ex in ds:
        msgs = _as_messages(ex["messages"])
        if not msgs or msgs[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in msgs):
            continue
        rows.append(
            {
                "prompt_id": ex["prompt_id"],
                "category": ex["category"],
                "prompt": format_dialog_prompt(msgs[:-1]),
                "context": dialog_plain(msgs[:-1]),
                "response": msgs[-1]["content"].strip(),
            }
        )
    df = pd.DataFrame(rows)
    val, test = train_test_split(
        df, train_size=N_VAL, stratify=df["category"], random_state=SEED, shuffle=True
    )
    return val.reset_index(drop=True), test.reset_index(drop=True)


def load_preference_pairs(n: int = N_PREF) -> pd.DataFrame:
    """ultrafeedback_binarized (train_prefs) üzerinden n örneklik alt küme."""
    from datasets import load_dataset

    ds = load_dataset(ULTRAFEEDBACK_ID, split="train_prefs")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    rows = []
    for ex in ds:
        prompt = ex["prompt"].strip()
        chosen = _last_assistant(ex["chosen"])
        rejected = _last_assistant(ex["rejected"])
        if not prompt or not chosen or not rejected:
            continue
        rows.append(
            {
                "raw_prompt": prompt,
                "prompt": format_prompt(prompt),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return pd.DataFrame(rows)


def _last_assistant(msgs) -> str:
    msgs = _as_messages(msgs)
    for m in reversed(msgs):
        if m["role"] == "assistant":
            return m["content"].strip()
    return ""


# --------------------------------------------------------------------------- #
# Simgeleştirme ve toplu birleştirme
# --------------------------------------------------------------------------- #
def tokenize_pairs(
    tok,
    prompts: list[str],
    responses: list[str],
    max_len: int = MAX_LEN,
    max_prompt_len: int = MAX_PROMPT_LEN,
) -> list[dict]:
    """
    Her (x, y) için input_ids / labels üretir. İstem simgeleri -100 ile maskelenir,
    yanıtın sonuna EOS eklenir. Uzun istemler sondan, uzun yanıtlar baştan korunarak kırpılır.
    `raw_lx` / `raw_ly` kırpma öncesi simge sayılarıdır.
    """
    p_all = tok(list(prompts), add_special_tokens=False)["input_ids"]
    r_all = tok(list(responses), add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    out = []
    for p, r in zip(p_all, r_all):
        r = list(r) + [eos]
        raw_lx, raw_ly = len(p), len(r)
        p = p[-max_prompt_len:]
        r = r[: max(1, max_len - len(p))]
        out.append(
            {
                "input_ids": p + r,
                "labels": [-100] * len(p) + r,
                "prompt_len": len(p),
                "raw_lx": raw_lx,
                "raw_ly": raw_ly,
            }
        )
    return out


class PadCollator:
    """input_ids / labels listelerini sağdan doldurarak tensör yığınına çevirir."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def parse_overrides(parser, overrides: dict, argv: list[str] | None = None):
    """
    Defansif bir argparse köprüsü: defter hücrelerinden `get_args(limit=2000)` gibi çağrılır,
    komut satırından ise `argv` ile. Bilinmeyen anahtar yazım hatalarını erken yakalar.
    """
    args = parser.parse_args(argv if argv is not None else [])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise AttributeError(f"Bilinmeyen parametre '{k}'. Geçerli olanlar: {sorted(vars(args))}")
        setattr(args, k, v)
    return args


def make_config(cls, **kwargs):
    """Sürümler arası API farklarına karşı: sınıfın kabul etmediği anahtarları atar."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[uyarı] {cls.__name__} şu argümanları desteklemiyor, atlandı: {dropped}")
    return cls(**{k: v for k, v in kwargs.items() if k in valid})
