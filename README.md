# gp-stratified-sft-selection

**GP Destekli Tabakalı SFT Veri Seçimi ve Çift Ölçekli (Proxy → Target) Hizalama**

Qwen2.5-0.5B-Base üzerinde **genetik programlama (DEAP)** ile bir SFT örnek-puanlama formülü ($f_{GP}^*$) evrimleştirilir;
formül, 20 anlamsal kümeden **tabakalı** 1.000 örnek seçer ve seçim **Qwen2.5-3B-Base** hedef modeline aktarılır.
Hedef modelde 5 SFT kıyas modeli ile iki hatlı tercih optimizasyonu (GRPO; DPO ve R-DPO) karşılaştırılır.

## Proje yapısı

```
gp-stratified-sft-selection/
├── src/                          # YALNIZCA fonksiyon/sınıf kütüphanesi (import edilir, çalıştırılmaz)
│   ├── common.py                 # sabitler, veri yükleyiciler, 4-bit NF4, VRAM temizliği, maliyet ölçümü
│   ├── features.py               # Aşama 1: bge-small gömme + KMeans(20) + 6 öznitelik (Qwen2.5-0.5B)
│   ├── gp_evolution.py           # Aşama 2: DEAP GP, önbellekli tabakalı vekil uygunluk, LoRA sıfırlamalı mikro-SFT
│   ├── alignment.py              # Aşama 3: 5 SFT + GRPO (harici ödül) + DPO + RDPOTrainer
│   ├── evaluation.py             # Aşama 4: üretim, D_KL, gpt-4o hakem, LC kazanma oranı, bootstrap GA
│   └── analysis.py               # akademik tablo/şekil yardımcıları (numaralı, 300 dpi + LaTeX)
├── notebooks/                    # raporlanabilir çıktıları üreten defterler
│   ├── 00_setup_and_data.ipynb
│   ├── 01_features_and_clustering.ipynb
│   ├── 02_gp_evolution.ipynb
│   ├── 03_target_sft_and_alignment.ipynb
│   └── 04_evaluation_and_judge.ipynb
├── Datasets/                     # (isteğe bağlı) yerel veri; defterler load_dataset ile Hub'dan da indirebilir
├── outputs/                      # çalıştırma sonrası üretilir (git'e dahil değil)
│   ├── features/ gp/ models/ generations/ metrics/ judge/
│   ├── figures/                  # fig_*.png (300 dpi) + fig_*.pdf
│   └── tables/                   # tab_*.csv + tab_*.tex
└── requirements.txt
```

`src/` modülleri komut satırı betiği değil kütüphanedir; her aşamanın girişi bir fonksiyondur:

| Aşama | Giriş fonksiyonu | Ek |
|---|---|---|
| 1 | `features.run_feature_extraction(limit=0, force=False)` | |
| 2 | `gp_evolution.run_gp_evolution(pop=16, gens=10, steps=60)` | `dry_run=True` GPU'suz mantık testi |
| 3 | `alignment.run_alignment(stage="sft"\|"grpo"\|"dpo"\|"rdpo"\|"all")` | `alignment.rdpo_logit_check()` |
| 4 | `evaluation.run_evaluation(stage="generate"\|"plots"\|"judge"\|"all")` | |

## Google Colab'da çalıştırma

1. Depoyu Drive'a yükleyin veya klonlayın (`MyDrive/gp-stratified-sft-selection`). Veri kümeleri `load_dataset` ile otomatik indirilir; `Datasets/` gerekmez.
2. Çalışma zamanı: **A100 80 GB** (önerilen) veya L4 24 GB. Defterleri **sırayla** çalıştırın (00 → 04).
3. İlk hücrede (00) `%pip install -r requirements.txt` satırının yorumunu kaldırın.
4. Defter 04'te `OPENAI_API_KEY` (Colab *Secrets* veya `getpass`) gerekir; diğerlerinde gerekmez.
5. Oturum kesilirse yeniden çalıştırın: SFT/DPO/GRPO adaptörleri, GP `fitness_cache` ve hakem yargıları diskten devam eder.

## Tasarım kararları ve notlar

* **Veri sızıntısı:** `no_robots` test bölümü kategoriye göre tabakalı iki yarıya bölünür — $D_{val\text{-}250}$ (GP uygunluğu) ve **ayrık** $D_{test\text{-}250}$ (nihai değerlendirme).
* **Vekil model tek sefer yüklenir;** her birey için LoRA ağırlıkları `reset_lora_parameters` ile sıfırlanır; aynı alt küme `fitness_cache` ile yeniden eğitilmez.
* **R-DPO**, TRL kaybını kopyalamadan referans log-olasılıklarını $-s\alpha|y|/\beta$ kadar kaydırarak uygulanır; özdeşlik `rdpo_logit_check()` ile sayısal olarak doğrulanır. İşaret `rdpo_sign` ile seçilir (+1: Park vd. 2024 Eş. 9; −1: proje notundaki ifade).
* **LC kazanma oranı:** uzunluk farkına göre lojistik düzeltme (AlpacaEval-LC ilkesi); %95 güven aralığı istem düzeyinde bootstrap, anlamlılık için işaret testi (+ Holm).
* **GRPO**, 10k tercih alt kümesinden `--grpo-prompts` (varsayılan 1.000) istemle çalışır; GPU bütçesine göre artırılabilir.
* Taban modeller: `Qwen/Qwen2.5-0.5B` ve `Qwen/Qwen2.5-3B` (Hugging Face'teki *Base* sürümleridir). Ortam değişkenleriyle (`PROXY_MODEL`, `TARGET_MODEL`, `REWARD_MODEL`) değiştirilebilir.
