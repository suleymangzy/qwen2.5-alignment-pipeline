"""
Modül 3 - Hedef modele (Qwen2.5-3B-Base) aktarım ve çift hatlı hizalama deneyleri.

Aşama A (--stage sft): 5 SFT kıyas modeli (4-bit QLoRA, 3 epoch, lr=2e-4)
    Model_SFT_Full | Model_SFT_Rand1k | Model_SFT_IFD1k | Model_SFT_GP_Top1k | Model_SFT_GP_Strat1k
Aşama B (--stage grpo): Model_IFD_GRPO, Model_GP_GRPO
    TRL GRPOTrainer + dondurulmuş, 4-bit nicemlenmiş Ray2333/GRM-Llama3-8B-rewardmodel-ft (r_phi)
Aşama C (--stage dpo):  Model_IFD_DPO, Model_GP_DPO        (trl.DPOTrainer)
Aşama D (--stage rdpo): Model_IFD_RDPO, Model_GP_RDPO      (RDPOTrainer = DPOTrainer + uzunluk düzenlemesi)

Her çalıştırma bitince adaptör diske yazılır, model/eğitici silinir ve VRAM boşaltılır; süre ve tepe VRAM
`outputs/metrics/training_costs.json` dosyasına, eğitim kayıp eğrileri `outputs/metrics/<model>_log.json`
dosyasına kaydedilir. Tamamlanmış modeller (--force verilmedikçe) atlanır; kesilen oturum kaldığı yerden sürer.

Kullanım (defter içinden):
    from alignment import run_alignment
    run_alignment(stage="sft")                       # 5 SFT modeli
    run_alignment(stage="align", models=["Model_GP_DPO"])
"""
from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from common import (
    ALIGNED_MODELS,
    ALL_MODELS,
    FEAT_DIR,
    GP_DIR,
    MAX_LEN,
    METRIC_DIR,
    MODEL_DIR,
    REWARD_MODEL,
    SEED,
    SFT_MODELS,
    TARGET_MODEL,
    bnb_config,
    compute_dtype,
    format_prompt,
    free_vram,
    gpu_total_gb,
    has_cuda,
    load_causal_lm,
    load_json,
    load_preference_pairs,
    load_tokenizer,
    lora_config,
    make_config,
    parse_overrides,
    save_json,
    set_seed,
    track_cost,
    _dtype_kwarg,
)

SUBSET_DIR = GP_DIR / "subsets"
SUBSET_FILES = {
    "Model_SFT_Rand1k": "rand1k.json",
    "Model_SFT_IFD1k": "ifd_top1k.json",
    "Model_SFT_GP_Top1k": "gp_top1k.json",
    "Model_SFT_GP_Strat1k": "gp_strat1k.json",
}


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def get_args(argv: list[str] | None = None, **overrides):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "grpo", "dpo", "rdpo", "align", "all"], default="all",
                    help="align = grpo+dpo+rdpo")
    ap.add_argument("--models", nargs="*", default=None, help="Yalnızca bu model adlarını çalıştır")
    ap.add_argument("--force", action="store_true", help="Var olan adaptörlerin üzerine yeniden eğit")
    ap.add_argument("--limit", type=int, default=0, help="Duman testi: her veri kümesinden ilk N örnek")
    ap.add_argument("--max-steps", type=int, default=-1, help="Duman testi: en çok N optimizasyon adımı")
    # SFT
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--full-epochs", type=float, default=None,
                    help="Model_SFT_Full için epoch (varsayılan: --epochs; 51.760 örnekte uzun sürer)")
    # DPO / R-DPO
    ap.add_argument("--dpo-lr", type=float, default=5e-5)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-epochs", type=float, default=1.0)
    ap.add_argument("--rdpo-alpha", type=float, default=0.05, help="R-DPO uzunluk düzenleme katsayısı alpha")
    ap.add_argument("--rdpo-sign", type=int, choices=[1, -1], default=1,
                    help="+1: Park vd. (2024) Eş. 9, logit'e +alpha(|yw|-|yl|) (uzunluk avantajını indirger). "
                         "-1: logit'e -alpha(|yw|-|yl|) (proje notundaki işaret)")
    # GRPO
    ap.add_argument("--grpo-prompts", type=int, default=1000, help="GRPO için kullanılan istem sayısı (10k tercih alt kümesinden)")
    ap.add_argument("--grpo-lr", type=float, default=2e-5)
    ap.add_argument("--grpo-beta", type=float, default=0.02)
    ap.add_argument("--grpo-generations", type=int, default=4)
    ap.add_argument("--grpo-max-completion", type=int, default=256)
    return parse_overrides(ap, {k.replace("-", "_"): v for k, v in overrides.items()}, argv)


def batch_plan(kind: str) -> tuple[int, int]:
    """(cihaz başı toplu, gradyan birikimi) -> etkin toplu 16. A100-80GB'ta büyük, L4-24GB'ta küçük."""
    big = gpu_total_gb() >= 60
    if kind == "sft":
        return (8, 2) if big else (4, 4)
    if kind == "pref":
        return (4, 4) if big else (2, 8)
    return (8, 2) if big else (4, 4)  # grpo (tamamlama sayısı)


def precision_flags() -> dict:
    dt = compute_dtype()
    return {"bf16": dt == torch.bfloat16 and has_cuda(), "fp16": dt == torch.float16 and has_cuda()}


def is_done(name: str) -> bool:
    return (MODEL_DIR / name / "adapter_config.json").exists()


def selected(names: list[str], args) -> list[str]:
    return [n for n in names if (not args.models or n in args.models)]


def dump_log(name: str, trainer) -> None:
    save_json(trainer.state.log_history, METRIC_DIR / f"{name}_log.json")


def cleanup(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    free_vram()


def save_adapter(model, name: str) -> Path:
    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(out, selected_adapters=["default"])  # "ref" kopyası kaydedilmez
    except TypeError:
        model.save_pretrained(out)
    shutil.rmtree(MODEL_DIR / f"_tmp_{name}", ignore_errors=True)  # Trainer'ın boş çıktı klasörü
    return out


def prepare_base(use_gc: bool = True):
    """3B tabanı 4-bit NF4 yükler ve k-bit eğitime hazırlar."""
    from peft import prepare_model_for_kbit_training

    model = load_causal_lm(TARGET_MODEL, quantize=True)
    if has_cuda():
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gc, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def load_policy_from_sft(sft_name: str):
    """4-bit 3B taban + eğitilebilir SFT adaptörü (hizalamanın başlangıç politikası)."""
    from peft import PeftModel

    if not is_done(sft_name):
        raise FileNotFoundError(f"{sft_name} bulunamadı; önce '--stage sft' çalıştırın.")
    base = prepare_base()
    return PeftModel.from_pretrained(base, MODEL_DIR / sft_name, is_trainable=True)


# --------------------------------------------------------------------------- #
# Aşama A: SFT
# --------------------------------------------------------------------------- #
def sft_indices(name: str, n_pool: int, args) -> np.ndarray:
    if name == "Model_SFT_Full":
        idx = np.arange(n_pool)
    else:
        path = SUBSET_DIR / SUBSET_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} yok; önce 02_gp_proxy_evolution.py çalıştırın.")
        idx = np.asarray(load_json(path), dtype=np.int64)
    return idx[: args.limit] if args.limit else idx


def run_sft(name: str, pool: pd.DataFrame, args) -> None:
    from trl import SFTConfig, SFTTrainer

    if is_done(name) and not args.force:
        print(f"[atla] {name} zaten var")
        return
    set_seed()
    idx = sft_indices(name, len(pool), args)
    sub = pool.iloc[idx]
    ds = Dataset.from_dict(
        {"prompt": [format_prompt(p) for p in sub["prompt"]], "completion": sub["response"].tolist()}
    ).shuffle(seed=SEED)
    epochs = args.full_epochs if (name == "Model_SFT_Full" and args.full_epochs) else args.epochs
    bs, accum = batch_plan("sft")
    print(f"\n##### {name}: {len(ds)} örnek, {epochs} epoch, etkin toplu={bs * accum}")

    tok = load_tokenizer(TARGET_MODEL)
    cfg = make_config(
        SFTConfig,
        output_dir=str(MODEL_DIR / f"_tmp_{name}"),
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10 if len(ds) < 5000 else 50,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        max_length=MAX_LEN,
        completion_only_loss=True,
        packing=False,
        seed=SEED,
        dataloader_num_workers=2,
        **precision_flags(),
    )
    with track_cost(name, "sft"):
        model = prepare_base()
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config())
        trainer.train()
        save_adapter(trainer.model, name)
        dump_log(name, trainer)
    cleanup(trainer, model)


# --------------------------------------------------------------------------- #
# Tercih verisi
# --------------------------------------------------------------------------- #
def load_pref_dataset(args, n: int | None = None) -> Dataset:
    cache = FEAT_DIR / "ultrafeedback_10k.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = load_preference_pairs()
        df.to_parquet(cache)
    if args.limit:
        df = df.iloc[: args.limit]
    if n:
        df = df.iloc[:n]
    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)


# --------------------------------------------------------------------------- #
# Aşama C/D: DPO ve R-DPO
# --------------------------------------------------------------------------- #
def build_rdpo_trainer_class():
    """
    R-DPO (Park vd., 2024):  L = -log sigma( beta*[log pi/pi_ref](yw) - beta*[log pi/pi_ref](yl) + s*alpha*(|yw| - |yl|) )

    Ek terim, TRL'in kayıp fonksiyonunu kopyalamadan, referans log-olasılıklarına kaydırılarak uygulanır:
        ref_w' = ref_w - (s*alpha/beta)*|yw| ,  ref_l' = ref_l - (s*alpha/beta)*|yl|
    => beta*(ref_l' - ref_w') = beta*(ref_l - ref_w) + s*alpha*(|yw| - |yl|).
    Referans log-olasılıkları `precompute_ref_log_probs=True` ile toplu içinde (inputs) geldiği için kayıp,
    TRL'in standart sigmoid DPO yolunu aynen kullanır; alpha=0 iken sonuç standart DPO ile özdeştir.
    """
    from trl import DPOTrainer

    class RDPOTrainer(DPOTrainer):
        def __init__(self, *args, length_alpha: float = 0.05, length_sign: int = 1, **kwargs):
            self.length_alpha = float(length_alpha)
            self.length_sign = int(length_sign)
            super().__init__(*args, **kwargs)
            if not getattr(self, "precompute_ref_logps", False):
                raise ValueError("RDPOTrainer için DPOConfig(precompute_ref_log_probs=True) gereklidir.")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            mask = inputs["completion_mask"]  # [2B, T] -> ilk yarı chosen, ikinci yarı rejected
            lens = mask.sum(dim=1).to(inputs["ref_chosen_logps"].dtype)
            len_w, len_l = lens.chunk(2, dim=0)
            k = self.length_sign * self.length_alpha / self.beta
            inputs["ref_chosen_logps"] = inputs["ref_chosen_logps"] - k * len_w
            inputs["ref_rejected_logps"] = inputs["ref_rejected_logps"] - k * len_l
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    return RDPOTrainer


def rdpo_logit_check(n: int = 10_000, beta: float = 0.1, alpha: float = 0.05, sign: int = 1, seed: int = SEED) -> dict:
    """
    R-DPO uygulamasının sayısal doğrulaması: referans kaydırma hilesi, tanımdaki
    -log sigma(beta*(dw - dl) + s*alpha*(|yw|-|yl|)) ifadesiyle birebir aynı kaybı verir mi?
    (dw, dl: log pi_theta/pi_ref farkları). Dönüş: iki kaybın ortalaması ve en büyük mutlak farkı.
    """
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    pol_w, pol_l = torch.randn(n, generator=g) * 20 - 100, torch.randn(n, generator=g) * 20 - 120
    ref_w, ref_l = torch.randn(n, generator=g) * 20 - 100, torch.randn(n, generator=g) * 20 - 120
    len_w, len_l = torch.randint(5, 400, (n,), generator=g).float(), torch.randint(5, 400, (n,), generator=g).float()

    direct = -F.logsigmoid(beta * ((pol_w - ref_w) - (pol_l - ref_l)) + sign * alpha * (len_w - len_l))
    k = sign * alpha / beta
    ref_w2, ref_l2 = ref_w - k * len_w, ref_l - k * len_l
    shifted = -F.logsigmoid(beta * ((pol_w - ref_w2) - (pol_l - ref_l2)))
    plain = -F.logsigmoid(beta * ((pol_w - ref_w) - (pol_l - ref_l)))
    return {"direct_mean": direct.mean().item(), "shifted_mean": shifted.mean().item(),
            "max_abs_diff": (direct - shifted).abs().max().item(), "plain_dpo_mean": plain.mean().item()}


def run_dpo(name: str, sft_name: str, rdpo: bool, args) -> None:
    from trl import DPOConfig, DPOTrainer

    if is_done(name) and not args.force:
        print(f"[atla] {name} zaten var")
        return
    set_seed()
    ds = load_pref_dataset(args)
    bs, accum = batch_plan("pref")
    print(f"\n##### {name} ({'R-DPO' if rdpo else 'DPO'}) <- {sft_name}: {len(ds)} çift, etkin toplu={bs * accum}")
    cfg = make_config(
        DPOConfig,
        output_dir=str(MODEL_DIR / f"_tmp_{name}"),
        num_train_epochs=args.dpo_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=args.dpo_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.dpo_beta,
        loss_type=["sigmoid"],
        max_length=MAX_LEN,
        precompute_ref_log_probs=True,  # referans = SFT adaptörünün kopyası ("ref"); R-DPO ile aynı yol
        precompute_ref_batch_size=bs * 2,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=SEED,
        **precision_flags(),
    )
    tok = load_tokenizer(TARGET_MODEL)
    with track_cost(name, "rdpo" if rdpo else "dpo"):
        model = load_policy_from_sft(sft_name)
        if rdpo:
            cls = build_rdpo_trainer_class()
            trainer = cls(model=model, args=cfg, train_dataset=ds.select_columns(["prompt", "chosen", "rejected"]),
                          processing_class=tok, length_alpha=args.rdpo_alpha, length_sign=args.rdpo_sign)
        else:
            trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds.select_columns(["prompt", "chosen", "rejected"]),
                                 processing_class=tok)
        trainer.train()
        save_adapter(trainer.model, name)
        dump_log(name, trainer)
    cleanup(trainer, model)


# --------------------------------------------------------------------------- #
# Aşama B: GRPO + dondurulmuş harici ödül modeli
# --------------------------------------------------------------------------- #
class FrozenRewardModel:
    """4-bit nicemlenmiş, dondurulmuş Ray2333/GRM-Llama3-8B-rewardmodel-ft  (r_phi(x, y) skaler ödül)."""

    def __init__(self, name: str = REWARD_MODEL, max_len: int = 1024, batch: int = 4):
        from transformers import AutoModelForSequenceClassification

        self.tok = load_tokenizer(name, padding_side="right")
        kwargs = _dtype_kwarg(compute_dtype())
        if has_cuda():
            kwargs.update(device_map={"": torch.cuda.current_device()}, quantization_config=bnb_config())
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, **kwargs).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.max_len, self.batch = max_len, batch
        self.device = next(self.model.parameters()).device
        self.__name__ = "grm_llama3_8b_reward"

    @torch.no_grad()
    def score(self, instructions: list[str], responses: list[str]) -> list[float]:
        texts = []
        for x, y in zip(instructions, responses):
            msgs = [{"role": "user", "content": x}, {"role": "assistant", "content": y if y.strip() else " "}]
            if self.tok.chat_template:
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False))
            else:
                texts.append(f"User: {x}\nAssistant: {y}")
        out: list[float] = []
        for s in range(0, len(texts), self.batch):
            enc = self.tok(texts[s : s + self.batch], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len, add_special_tokens=False).to(self.device)
            out.extend(self.model(**enc).logits[:, 0].float().cpu().tolist())
        return out

    def __call__(self, prompts, completions, raw_prompt=None, **kwargs) -> list[float]:
        instr = list(raw_prompt) if raw_prompt is not None else list(prompts)
        return self.score(instr, list(completions))

    def close(self):
        del self.model
        free_vram()


def run_grpo(name: str, sft_name: str, args) -> None:
    from trl import GRPOConfig, GRPOTrainer

    if is_done(name) and not args.force:
        print(f"[atla] {name} zaten var")
        return
    set_seed()
    n = args.limit or args.grpo_prompts
    ds = load_pref_dataset(args)
    tok = load_tokenizer(TARGET_MODEL, padding_side="left")
    keep = [i for i, p in enumerate(ds["prompt"]) if len(tok(p)["input_ids"]) <= 256][:n]
    ds = ds.select(keep).select_columns(["prompt", "raw_prompt"])
    bs, accum = batch_plan("grpo")
    G = args.grpo_generations
    bs = max(G, (bs // G) * G)  # tamamlama sayısı num_generations'a bölünebilmeli
    print(f"\n##### {name} (GRPO) <- {sft_name}: {len(ds)} istem, G={G}, etkin tamamlama/adım={bs * accum}")
    cfg = make_config(
        GRPOConfig,
        output_dir=str(MODEL_DIR / f"_tmp_{name}"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=args.grpo_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.grpo_beta,
        num_generations=G,
        max_completion_length=args.grpo_max_completion,
        temperature=0.9,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=SEED,
        **precision_flags(),
    )
    with track_cost(name, "grpo"):
        model = load_policy_from_sft(sft_name)
        reward = FrozenRewardModel()
        trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=cfg, train_dataset=ds, processing_class=tok)
        trainer.train()
        save_adapter(trainer.model, name)
        dump_log(name, trainer)
    reward.close()
    cleanup(trainer, model, reward)


# --------------------------------------------------------------------------- #
def run_alignment(args=None, **overrides) -> dict:
    """Seçilen aşamaları (sft/grpo/dpo/rdpo) sırayla çalıştırır; tamamlanan adaptörleri atlar."""
    args = args or get_args(**overrides)
    set_seed()
    stages = {"sft", "grpo", "dpo", "rdpo"} if args.stage == "all" else (
        {"grpo", "dpo", "rdpo"} if args.stage == "align" else {args.stage})

    if "sft" in stages:
        pool = pd.read_parquet(FEAT_DIR / "pool.parquet")
        for name in selected(SFT_MODELS, args):
            run_sft(name, pool, args)
    for name, (src, method) in ALIGNED_MODELS.items():
        if method in stages and selected([name], args):
            if method == "grpo":
                run_grpo(name, src, args)
            else:
                run_dpo(name, src, method == "rdpo", args)
    print("\n[tamam] adaptörler:", MODEL_DIR)
    return {"trained": [n for n in ALL_MODELS if is_done(n)], "costs": load_json(METRIC_DIR / "training_costs.json", {})}
