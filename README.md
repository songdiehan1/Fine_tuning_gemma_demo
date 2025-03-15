# Fine-tuning Gemma-3-4B with LoRA on Google Colab

This project fine-tunes the Gemma-3-4B model using LoRA (Low-Rank Adaptation), optimized with Unsloth + PEFT + SFT. The training is performed on Google Colab with 4-bit quantization (QLoRA) to reduce VRAM usage, using high-quality Chinese Zhihu dataset.

## 📌 Key Features

⚡ Optimized with Unsloth, making fine-tuning 2-4x faster than traditional methods.

🎯 Uses LoRA fine-tuning, reducing computational cost.

📖 Trains on high-quality Zhihu SFT dataset.

✅ Compatible with Google Colab for easy execution.

## 🚀 Quick Start

📌 Click the link below to open the Colab notebook and start training immediately:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O3WAp2oBoayjeMwB57RLoA5SVUcjcasM)


## 🛠️ Installation
Run the following commands in Google Colab:
```bash
# Install dependencies
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

## 📂 Dataset Preparation
This project uses the DeepSeek SFT dataset and selects high-score conversations from Zhihu.
```bash
dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", streaming=True)
filtered_dataset = dataset.filter(lambda example: example['repo_name'] == 'zhihu/zhihu_score9.0-10_clean_v10')
```

## 🤖 Load Gemma-3-4B and Apply LoRA Fine-Tuning
```bash
# Load the Gemma-3-4B model (optimized with Unsloth)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,  # Enable 4-bit quantization for lower VRAM usage
    load_in_8bit=False,
    full_finetuning=False,
)
# Apply LoRA fine-tuning
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=23,
)
```
### 📌 Notes:
✅LoRA dimension r=8 reduces parameter updates while maintaining efficiency.

✅4-bit quantization enabled for memory efficiency.

# 🔥 Fine-Tuning Process
```bash
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=23,
        report_to="none",
    ),
)
trainer.train()
```

