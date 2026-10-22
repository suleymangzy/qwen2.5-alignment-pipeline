# LLM Hizalama Deneyi: SFT → RLHF (PPO) → DPO — Qwen2.5-0.5B

Bu proje, deneysel kurguda tanımlanan hizalama (alignment) boru hattını
(pipeline) uygular: **Qwen/Qwen2.5-0.5B** taban modeli üzerinde Gözetimli
İnce Ayar (SFT) → Ödül Modeli + PPO (RLHF, çoklu-LoRA-adaptör yöntemi) → DPO.

## Klasör Yapısı

```
Uygulama/
├── Datasets/                               # Yerel veri kümeleri (önceden mevcut)
│   ├── HuggingFaceH4-no_robots/            # SFT gösterim verisi (Rajani vd., 2023)
│   └── Anthropichh-rlhf/                   # Tercih verisi (Bai vd., 2022)
├── src/
│   └── alignment_pipeline.py               # Tüm fonksiyonların bulunduğu ortak kütüphane
├── notebooks/
│   ├── 01_Data_Analysis.ipynb
│   ├── 02_SFT_Training.ipynb
│   ├── 03_RLHF_Training.ipynb              # Ödül Modeli + PPO (çoklu-adaptör)
│   ├── 04_DPO_Training.ipynb
│   ├── 05_Algorithm_Comparison_Quantitative.ipynb
│   └── 06_Algorithm_Comparison_Qualitative.ipynb   # LLM-as-a-Judge
└── outputs/                                 # Çalıştırma sonrası üretilir (adaptörler, loglar, grafikler)
```

## Veri Kümeleri

`Datasets/` klasörü büyük dosya boyutları nedeniyle (bir hh-rlhf dosyası
GitHub'ın 100MB tekil dosya sınırını aşmaktadır) bu depoya **dahil
edilmemiştir** (`.gitignore`). Çalıştırmadan önce aşağıdaki iki veri kümesini
Hugging Face Hub'dan indirip `Datasets/` altına, [src/alignment_pipeline.py](src/alignment_pipeline.py)
içindeki yükleyicilerin beklediği klasör yapısıyla yerleştirin:

- `Datasets/HuggingFaceH4-no_robots/` ← [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (`train_sft`, `test_sft` parquet dosyaları)
- `Datasets/Anthropichh-rlhf/<subset>/` ← [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (`harmless_base`, `helpful_base`, `helpful_online`, `helpful_rejection_sampled` alt kümeleri, `train.jsonl`/`test.jsonl`)

## Google Colab Pro Üzerinde Çalıştırma

1. `Uygulama/` klasörünün tamamını (özellikle `Datasets/` ve `src/`) Google
   Drive'ınıza yükleyin, örneğin `MyDrive/Uygulama`.
2. Her defterin ilk kod hücresindeki `PROJECT_ROOT` değişkenini kendi Drive
   yolunuza göre güncelleyin (varsayılan: `/content/drive/MyDrive/Uygulama`).
3. Defterleri sırasıyla çalıştırın:
   `01_Data_Analysis` → `02_SFT_Training` → `03_RLHF_Training` →
   `04_DPO_Training` → `05_Algorithm_Comparison_Quantitative` →
   `06_Algorithm_Comparison_Qualitative`.
4. Çalışma zamanı türü olarak bir GPU (A100 veya L4) seçin
   (Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU).

Her defter, `src/alignment_pipeline.py` modülünü `sys.path`'e ekleyip
`import alignment_pipeline as ap` ile içe aktarır; bu nedenle `src/` klasörü
mutlaka `notebooks/` ile aynı üst klasörde (`Uygulama/`) bulunmalıdır.

## Yazılım Sürümleri (Pinned)

API kararlılığı için (özellikle TRL'nin klasik `PPOTrainer` +
`AutoModelForCausalLMWithValueHead` çoklu-adaptör RL özelliği), aşağıdaki
sürümler sabitlenmiştir:

```
transformers==4.44.2   peft==0.11.1        bitsandbytes==0.43.1
trl==0.9.6              accelerate==0.32.1  datasets==2.20.0
```

## Boru Hattı Özeti

| Aşama | Defter | Yöntem | Çıktı |
|---|---|---|---|
| Veri Analizi | `01_Data_Analysis` | — | Gözlemler, grafikler |
| SFT | `02_SFT_Training` | `trl.SFTTrainer` + LoRA | `Model_SFT` (`outputs/sft/merged_model`) |
| Ödül Modeli | `03_RLHF_Training` (Adım A) | `trl.RewardTrainer` + LoRA (SEQ_CLS) | `outputs/reward_model/adapter` |
| PPO (RLHF) | `03_RLHF_Training` (Adım B) | `trl.PPOTrainer` (çoklu-adaptör) | `Model_RLHF` (`outputs/ppo/ppo_adapter`) |
| DPO | `04_DPO_Training` | `trl.DPOTrainer` | `Model_DPO` (`outputs/dpo/adapter`) |
| Nicel Karşılaştırma | `05_Algorithm_Comparison_Quantitative` | Kayıp eğrileri, yanıt uzunluğu, $D_{KL}$ | Grafikler, özet tablo |
| Nitel Karşılaştırma | `06_Algorithm_Comparison_Qualitative` | LLM-as-a-Judge (GPT-4o / Llama-3.1-70B-Instruct) | Win-Rate matrisi |

## Üretim Senaryosu (Opsiyonel)

Bütçe elverdiğinde, `alignment_pipeline.ExperimentConfig.production_model_id`
(`Qwen/Qwen2.5-7B`) kullanılarak aynı fonksiyonlarla LoRA ya da Tam İnce Ayar
(Full Fine-Tuning) uygulanabilir; sadece defterlerdeki `config.base_model_id`
yerine `config.production_model_id` geçirilmesi ve donanıma göre
batch/gradient-accumulation değerlerinin güncellenmesi yeterlidir.
