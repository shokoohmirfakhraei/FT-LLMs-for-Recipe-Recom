# 1.1 Import Libraries and Load the Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display options
pd.set_option('display.max_columns', None)

# Load dataset
df = pd.read_csv("/content/pp_recipes.csv", engine='python', sep=',', on_bad_lines='skip')

# Preview
print(df.shape)
df.head()

"""# 1.2 Basic Information and Missing Values"""

# Overview
df.info()

# Missing value summary
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

"""# 1.3 Summary Statistics"""

# Focus on key numeric features
numeric_cols = [
    'calories [cal]',
    'protein [g]',
    'sugars [g]',
    'sodium [mg]',
    'duration'
]

# Generate descriptive statistics
summary = df[numeric_cols].describe()

# Display the summary table
print("Summary statistics for key nutritional features:\n")
print(summary)

"""# 1.4 Visualize Nutritional Distributions"""

print(df.columns)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Correct column names from your DataFrame
nutrients = ['calories [cal]', 'protein [g]', 'sodium [mg]', 'sugars [g]']

plt.figure(figsize=(12,8))

for i, col in enumerate(nutrients, 1):
    plt.subplot(2, 2, i)
    sns.histplot(np.log1p(df[col]), bins=50, kde=True)
    plt.title(f"Distribution of {col} [log scale]")
    plt.xlabel(f"log({col})")
    plt.ylabel("Number of Recipes")

plt.tight_layout()
plt.show()

"""# 1.5 Quick vs Long Recipes"""

plt.figure(figsize=(8,5))
sns.histplot(df['duration'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Recipe Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
plt.show()

quick = (df['duration'] <= 30).sum()
long  = (df['duration'] > 30).sum()
print(f"Quick recipes (≤30 min): {quick:,}")
print(f"Long recipes (>30 min): {long:,}")

"""# 1.5 Health Categories (derived from nutri_score)"""

df['nutri_score'].describe()

def health_category(score):
    if score <= 0.2:
        return "Healthy"
    elif score <= 0.5:
        return "Moderate"
    else:
        return "Less Healthy"

df['health_category'] = df['nutri_score'].apply(health_category)

plt.figure(figsize=(6,4))
sns.countplot(y='health_category', data=df,
              order=df['health_category'].value_counts().index,
              palette='viridis')
plt.title("Derived Health Categories from Nutri-Score")
plt.xlabel("Number of Recipes")
plt.ylabel("Health Category")
plt.show()

"""# 1.6 Top Tags"""

from collections import Counter
import ast

def extract_list_values(series):
    all_items = []
    for val in series.dropna():
        try:
            items = ast.literal_eval(val)
            if isinstance(items, list):
                all_items.extend(items)
        except:
            continue
    return all_items

# Extract top tags
tag_list = extract_list_values(df['tags'])
top_tags = Counter(tag_list).most_common(15)

plt.figure(figsize=(10,5))
sns.barplot(x=[x[1] for x in top_tags], y=[x[0] for x in top_tags])
plt.title("Top 15 Tags")
plt.xlabel("Frequency")
plt.ylabel("Tag")
plt.show()

"""# 1.7 Top Ingredients"""

ing_list = extract_list_values(df['ingredient_food_kg_names'])
top_ing = Counter(ing_list).most_common(15)

plt.figure(figsize=(10,5))
sns.barplot(x=[x[1] for x in top_ing], y=[x[0] for x in top_ing], color='coral')
plt.title("Top 15 Most Common Ingredients")
plt.xlabel("Frequency")
plt.ylabel("Ingredient")
plt.show()

"""# 1.8 Protein vs Calories Correlation"""

plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x='protein [g]', y='calories [cal]', alpha=0.6)
plt.title("Protein vs Calories")
plt.xlabel("Protein [g]")
plt.ylabel("Calories [cal]")
plt.show()

corr = df['protein [g]'].corr(df['calories [cal]'])
print(f"Correlation between Protein and Calories: {corr:.2f}")

# ============================================================
# STEP 2 — Unified Instruction Dataset Generation
# ============================================================
import pandas as pd, json, random, ast, re, numpy as np
from typing import List, Dict, Any

# ------------------------------------------------------------
# 1️⃣ Load dataset
# ------------------------------------------------------------
CSV_PATH = "/content/pp_recipes.csv"
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip", low_memory=True)
print(f"Loaded {len(df):,} recipes")

# ------------------------------------------------------------
# 2️⃣ Helpers
# ------------------------------------------------------------
def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except: return default

def safe_int(val, default=None):
    try:
        if pd.isna(val): return default
        return int(float(val))
    except: return default

def parse_ingredients_field(x):
    """Parse list-like or dict-like ingredient fields."""
    if pd.isna(x): return []
    if isinstance(x, dict):
        out = []
        for cat, items in x.items():
            for item in items:
                if isinstance(item, tuple): out.append(item[0])
                else: out.append(str(item))
        return out
    if isinstance(x, list): return [str(i).strip() for i in x]
    s = str(x)
    for p in (json.loads, ast.literal_eval):
        try:
            val = p(s)
            if isinstance(val, list): return [str(i).strip() for i in val]
        except: pass
    return [t.strip() for t in re.split(r"[;,|]", s) if t.strip()]

# ------------------------------------------------------------
# 3️⃣ Single-recipe variant generator
# ------------------------------------------------------------
def generate_instruction_variants(recipe: Dict[str, Any]) -> List[Dict[str, str]]:
    variants = []

    title = recipe.get("title", "Unknown Recipe")
    calories = safe_float(recipe.get("calories [cal]"))
    protein  = safe_float(recipe.get("protein [g]"))
    sodium   = safe_float(recipe.get("sodium [mg]"))
    fiber    = safe_float(recipe.get("dietaryFiber [g]"))
    fat      = safe_float(recipe.get("totalFat [g]"))
    carbs    = safe_float(recipe.get("totalCarbohydrate [g]"))
    duration = safe_int(recipe.get("duration"))
    serves   = safe_int(recipe.get("serves"), 1)
    ingredients = parse_ingredients_field(recipe.get("ingredient_food_kg_names") or recipe.get("ingredients"))

    main_ing = ingredients[0] if ingredients else "ingredients"

    # calorie-based
    if calories > 0:
        limit = 400 if calories < 400 else int(calories)
        variants.append({
            "instruction": f"Suggest a {main_ing} recipe under {limit} calories.",
            "input": "",
            "output": f"{title} – {calories:.1f} kcal, {protein:.1f}g protein, ready in {duration or '?'} min."
        })

    # time-based
    if duration and duration > 0:
        variants.append({
            "instruction": f"Recommend a quick recipe that takes less than {duration + 10} minutes.",
            "input": "",
            "output": f"{title} – {duration} min, {calories:.0f} kcal."
        })

    # high-protein
    if protein >= 20:
        variants.append({
            "instruction": f"Find a high-protein recipe with at least {int(protein - 2)}g protein.",
            "input": "",
            "output": f"{title} – {protein:.1f}g protein, {calories:.0f} kcal."
        })

    # low-sodium
    if sodium > 0:
        variants.append({
            "instruction": f"List a low-sodium recipe under {int(sodium + 50)}mg sodium.",
            "input": "",
            "output": f"{title} – {sodium:.1f}mg sodium, {calories:.0f} kcal."
        })

    # high-fiber
    if fiber >= 5:
        variants.append({
            "instruction": f"Recommend a high-fiber recipe with at least {int(fiber)}g fiber.",
            "input": "",
            "output": f"{title} – {fiber:.1f}g fiber, {calories:.0f} kcal, {protein:.1f}g protein."
        })

    # ingredient-based
    if len(ingredients) >= 2:
        ing1, ing2 = ingredients[0], ingredients[1]
        variants.append({
            "instruction": f"Suggest a recipe using {ing1} and {ing2}.",
            "input": "",
            "output": f"{title} – features {ing1} and {ing2}, {calories:.0f} kcal."
        })

    # balanced
    if 300 <= calories <= 600 and 10 <= protein <= 25:
        variants.append({
            "instruction": "Suggest a balanced meal with moderate calories and good protein.",
            "input": "",
            "output": f"{title} – {calories:.0f} kcal, {protein:.1f}g protein, {fat:.1f}g fat."
        })

    return variants

# ------------------------------------------------------------
# 4️⃣ Multi-recipe and combo queries
# ------------------------------------------------------------
def format_multi(subset, query_desc, limit=3):
    subset=subset.dropna(subset=["title"])[:limit]
    if subset.empty:
        return f"No recipes found for {query_desc}."
    lines=[]
    for _,r in subset.iterrows():
        lines.append(f"- {r['title']} ({r['calories [cal]']:.0f} kcal, {r['protein [g]']:.0f}g protein, {int(r['duration']) if pd.notna(r['duration']) else '?'} min)")
    return "\n".join(lines)

def generate_multi_recipe_queries(df: pd.DataFrame) -> List[Dict[str,str]]:
    pairs=[]

    # low-calorie
    lowcal=df[df["calories [cal]"]<400]
    if not lowcal.empty:
        pairs.append({
            "instruction":"List three low-calorie recipes under 400 kcal.",
            "input":"",
            "output":format_multi(lowcal,"low-calorie")
        })

    # quick
    quick=df[df["duration"].notna() & (df["duration"]<20)]
    if not quick.empty:
        pairs.append({
            "instruction":"List three quick recipes under 20 minutes.",
            "input":"",
            "output":format_multi(quick,"quick")
        })

    # high-protein
    highp=df[df["protein [g]"]>=20]
    if not highp.empty:
        pairs.append({
            "instruction":"List three high-protein recipes (≥20g protein).",
            "input":"",
            "output":format_multi(highp,"high-protein")
        })

    # low-sodium
    lows=df[df["sodium [mg]"]<60]
    if not lows.empty:
        pairs.append({
            "instruction":"List three low-sodium recipes (less than 60mg sodium).",
            "input":"",
            "output":format_multi(lows,"low-sodium")
        })

    # high-fiber
    if "dietaryFiber [g]" in df.columns:
        hf=df[df["dietaryFiber [g]"]>=5]
        if not hf.empty:
            pairs.append({
                "instruction":"List three high-fiber recipes (≥5g fiber).",
                "input":"",
                "output":format_multi(hf,"high-fiber")
            })

    # combo filter
    combo=df[(df["calories [cal]"]<=450)&(df["protein [g]"]>=15)&(df["duration"]<=30)]
    if not combo.empty:
        pairs.append({
            "instruction":"List three healthy high-protein recipes under 450 kcal and 30 minutes.",
            "input":"",
            "output":format_multi(combo,"combo")
        })

    return pairs

# ------------------------------------------------------------
# 5️⃣ Dataset creation
# ------------------------------------------------------------
def create_finetuning_dataset(df: pd.DataFrame, max_recipes=1000) -> List[Dict[str,str]]:
    df_clean=df.dropna(subset=["title","calories [cal]"])
    df_clean=df_clean[df_clean["calories [cal]"]>0]
    if len(df_clean)>max_recipes:
        df_clean=df_clean.sample(n=max_recipes,random_state=42)

    all_examples=[]
    for _,row in df_clean.iterrows():
        all_examples.extend(generate_instruction_variants(row.to_dict()))

    all_examples.extend(generate_multi_recipe_queries(df_clean))
    random.shuffle(all_examples)
    return all_examples

# ------------------------------------------------------------
# 6️⃣ Generate + Save
# ------------------------------------------------------------
examples=create_finetuning_dataset(df,max_recipes=1000)
path="/content/instruction_dataset.jsonl"

with open(path,"w",encoding="utf-8") as f:
    for ex in examples:
        json.dump(ex,f,ensure_ascii=False)
        f.write("\n")

print(f"✅ Created {len(examples)} instruction–response pairs")
print(f"📁 Saved to: {path}")

# preview a few
for s in random.sample(examples,5):
    print("\n--- Example ---")
    print("Instruction:", s["instruction"])
    print("Response:\n", s["output"])

# ============================================================
# STEP 3 — Data Formatting for Fine-Tuning  (with prompt cleanup)
# ============================================================
import json, random, re
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# 1️⃣ Load unified dataset
# ------------------------------------------------------------
src_path = Path("/content/instruction_dataset.jsonl")
assert src_path.exists(), "❌ Run Step 2.5 first to create instruction_dataset.jsonl"

data = []
with open(src_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(data):,} instruction–response pairs")

# ------------------------------------------------------------
# 2️⃣ Basic cleaning
# ------------------------------------------------------------
df = pd.DataFrame(data)
df = df.dropna(subset=["instruction", "output"])
df = df[df["instruction"].str.strip() != ""]
df = df[df["output"].str.strip() != ""]
print(f"After cleaning: {len(df):,} usable examples")

# ------------------------------------------------------------
# 3️⃣ 🧹 Clean numeric thresholds for more natural prompts
# ------------------------------------------------------------
def clean_instruction(text: str) -> str:
    """
    Round numeric values in instructions to friendlier numbers.
      2011 -> 2000
      669  -> 700
      23   -> 20
    """
    def _round_num(match):
        num = int(match.group())
        if num < 50:
            return str(int(round(num / 5.0) * 5))       # nearest 5
        elif num < 1000:
            return str(int(round(num / 50.0) * 50))     # nearest 50
        else:
            return str(int(round(num, -2)))             # nearest 100
    return re.sub(r"\b\d{2,4}\b", _round_num, text)

df["instruction"] = df["instruction"].apply(clean_instruction)

# ------------------------------------------------------------
# 4️⃣ Shuffle and split (80/10/10)
# ------------------------------------------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

splits = {
    "train": df.iloc[:train_end],
    "validation": df.iloc[train_end:val_end],
    "test": df.iloc[val_end:]
}

# ------------------------------------------------------------
# 5️⃣ Save each split as JSONL
# ------------------------------------------------------------
out_dir = Path("/content/splits")
out_dir.mkdir(parents=True, exist_ok=True)

for name, subset in splits.items():
    path = out_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            obj = {
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(subset):,} → {path}")

# ------------------------------------------------------------
# 6️⃣ Sanity-check samples
# ------------------------------------------------------------
for name, subset in splits.items():
    print(f"\n--- {name.upper()} SAMPLE ---")
    sample = subset.sample(2, random_state=1)
    for _, r in sample.iterrows():
        print("Instruction:", r["instruction"])
        print("Response:", r["output"])
        print()

# ============================================================
# STEP 4 — Baseline Evaluation (Unsloth, Memory-Optimized)
# ============================================================
!pip install --no-deps bitsandbytes accelerate trl peft datasets evaluate rouge_score unsloth -q

from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load as load_metric
import torch, json, os

# ------------------------------------
# 1️⃣ Model setup (T4-safe)
# ------------------------------------
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 1024     # reduce to save VRAM
dtype = None
load_in_4bit = True

print("🚀 Loading model (Unsloth handles memory automatically)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

# ------------------------------------
# 2️⃣ Load your test split
# ------------------------------------
dataset = load_dataset("json", data_files={"test": "/content/splits/test.jsonl"})["test"]

prompt_template = """
### Human:
{}

### Assistant:
{}"""

EOS_TOKEN = "</s>"

# ------------------------------------
# 3️⃣ Generate baseline predictions
# ------------------------------------
def generate_predictions(ds, model, tokenizer, max_new_tokens=96):
    preds, refs = [], []
    for ex in tqdm(ds, desc="Generating baseline outputs"):
        instr, ref = ex["instruction"], ex["output"]
        inputs = tokenizer(
            [prompt_template.format(instr, "")],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        response = text.split("### Assistant:")[-1].strip()
        preds.append(response)
        refs.append(ref)
    return preds, refs

# ------------------------------------
# 4️⃣ Compute baseline metrics
# ------------------------------------
rouge = load_metric("rouge")
bleu = load_metric("bleu")

print("\n🧪 Running baseline generation...")
preds, refs = generate_predictions(dataset, model, tokenizer)

scores = {
    "rouge": rouge.compute(predictions=preds, references=refs),
    "bleu":  bleu.compute(predictions=preds, references=[[r] for r in refs]),
}

os.makedirs("/content/results", exist_ok=True)
with open("/content/results/baseline_metrics.json", "w") as f:
    json.dump(scores, f, indent=2)

print("\n✅ Baseline metrics saved → /content/results/baseline_metrics.json")
print("ROUGE:", scores["rouge"])
print("BLEU:", scores["bleu"])

# ============================================================
# STEP 5 — Fine-Tuning TinyLlama (FP32, tokenization-safe)
# ============================================================

!pip uninstall -y bitsandbytes triton -q
!pip install -U transformers==4.41.2 accelerate==0.30.1 trl==0.9.6 \
               peft==0.10.0 datasets evaluate rouge_score -q

# --- Imports ---
import os, json, gc, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from evaluate import load as load_metric
from tqdm import tqdm

# --- Safe CUDA environment ---
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- Paths ---
DATA_DIR = "/content/splits"
OUT_DIR  = "/content/outputs/recipe_lora"
RES_DIR  = "/content/results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# --- Model / runtime setup ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_seq_length = 512
random_state = 3407
device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

prompt_template = """### Human:
{}
### Assistant:
{}"""
EOS_TOKEN = "</s>"

# ============================================================
# 1️⃣  Load dataset
# ============================================================
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/validation.jsonl",
    },
)

# --- Formatting ---
def formatting_prompts_func(example):
    instr, out = example["instruction"], example["output"]
    # handle both single strings and lists
    if isinstance(instr, list):
        instr = " ".join([str(i) for i in instr])
    if isinstance(out, list):
        out = " ".join([str(o) for o in out])
    return {"text": prompt_template.format(instr, out) + EOS_TOKEN}

dataset = dataset.map(formatting_prompts_func)

# --- Filter out empty examples ---
dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
train_dataset = dataset["train"]
val_dataset   = dataset["validation"]

print(f"✅ Train examples: {len(train_dataset)}, Validation: {len(val_dataset)}")

# ============================================================
# 2️⃣  Tokenizer setup (safe padding & truncation)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"
tokenizer.model_max_length = max_seq_length

print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset   = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

# ============================================================
# 3️⃣  Load model (full precision)
# ============================================================
print("🚀 Loading TinyLlama in float32 …")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device_map,
)
model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = False

# ============================================================
# 4️⃣  Attach LoRA adapters
# ============================================================
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print("✅ LoRA adapters attached.")

# ============================================================
# 5️⃣  Trainer configuration (epoch-based, fp32)
# ============================================================
sft_config = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    num_train_epochs=1,
    logging_steps=5,
    report_to="none",
    seed=random_state,
    max_seq_length=max_seq_length,
    fp16=False, bf16=False,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ============================================================
# 6️⃣  Training
# ============================================================
print("🚀 Starting fine-tuning …")
trainer.train()
print("✅ Training completed!")

# ============================================================
# 7️⃣  Evaluation
# ============================================================
rouge = load_metric("rouge")
bleu  = load_metric("bleu")

def generate_predictions(ds, model, tokenizer, max_new_tokens=96):
    preds, refs = [], []
    for ex in tqdm(ds, desc="Evaluating fine-tuned outputs"):
        inputs = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        ref = tokenizer.decode(ex["labels"], skip_special_tokens=True)
        inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 temperature=0.7,
                                 top_p=0.9,
                                 do_sample=True)
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        preds.append(text)
        refs.append(ref)
    return preds, refs

val_preds, val_refs = generate_predictions(val_dataset, model, tokenizer)
val_scores = {
    "rouge": rouge.compute(predictions=val_preds, references=val_refs),
    "bleu":  bleu.compute(predictions=val_preds, references=[[r] for r in val_refs]),
}
with open(f"{RES_DIR}/fine_tuned_metrics.json","w") as f:
    json.dump(val_scores,f,indent=2)
print("✅ Metrics saved →",f"{RES_DIR}/fine_tuned_metrics.json")

# ============================================================
# 8️⃣  Save + Drive backup
# ============================================================
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("✅ Model saved at",OUT_DIR)

from google.colab import drive
drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/llama_recipe_project"
!cp -r /content/outputs/recipe_lora "/content/drive/MyDrive/llama_recipe_project/"
!cp -r /content/results "/content/drive/MyDrive/llama_recipe_project/"
print("✅ Backed up to Drive → MyDrive/llama_recipe_project/")

# ============================================================
# STEP 6 — Evaluation of the fine-tuned model
# ============================================================

from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import torch, json, os

# --- Paths ---
DATA_DIR = "/content/splits"
RES_DIR  = "/content/results"
os.makedirs(RES_DIR, exist_ok=True)

max_seq_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Load validation dataset ---
dataset = load_dataset("json", data_files={"validation": f"{DATA_DIR}/validation.jsonl"})
val_dataset = dataset["validation"]

# --- Load evaluation metrics ---
rouge = load_metric("rouge")
bleu  = load_metric("bleu")

# --- Generate predictions and collect references ---
def generate_predictions(ds, model, tokenizer, max_new_tokens=96):
    preds, refs = [], []
    for ex in tqdm(ds, desc="Evaluating fine-tuned outputs"):
        instr = ex.get("instruction") or ""
        ref   = ex.get("output") or ""
        if not instr.strip():
            continue

        # Rebuild the same prompt format used in training
        prompt = f"### Human:\n{instr}\n\n### Assistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True,
                           max_length=max_seq_length).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode and extract assistant text only
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        response = text.split("### Assistant:")[-1].split("###")[0].strip()

        preds.append(response)
        refs.append(ref.strip())

    return preds, refs

# --- Run evaluation ---
val_preds, val_refs = generate_predictions(val_dataset, model, tokenizer)

val_scores = {
    "rouge": rouge.compute(predictions=val_preds, references=val_refs),
    "bleu":  bleu.compute(predictions=val_preds, references=[[r] for r in val_refs]),
}

# --- Save metrics ---
with open(f"{RES_DIR}/fine_tuned_metrics.json", "w") as f:
    json.dump(val_scores, f, indent=2)

print("\n✅ Evaluation completed successfully!")
print("ROUGE:", val_scores["rouge"])
print("BLEU:", val_scores["bleu"])

# ============================================================
# STEP 7 — Final Evaluation on the Test Split
# ============================================================

from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import torch, json, os
from datetime import datetime

# --- Paths ---
DATA_DIR = "/content/splits"
RES_DIR  = "/content/results"
os.makedirs(RES_DIR, exist_ok=True)

max_seq_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Load test dataset ---
dataset = load_dataset("json", data_files={"test": f"{DATA_DIR}/test.jsonl"})
test_dataset = dataset["test"]
print(f"✅ Loaded test split with {len(test_dataset)} examples")

# --- Load metrics ---
rouge = load_metric("rouge")
bleu  = load_metric("bleu")

# --- Generate predictions ---
def generate_predictions(ds, model, tokenizer, max_new_tokens=96):
    preds, refs = [], []
    for ex in tqdm(ds, desc="Evaluating on test data"):
        instr = ex.get("instruction") or ""
        ref   = ex.get("output") or ""
        if not instr.strip():
            continue

        prompt = f"### Human:\n{instr}\n\n### Assistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True,
                           max_length=max_seq_length).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        response = text.split("### Assistant:")[-1].split("###")[0].strip()
        preds.append(response)
        refs.append(ref.strip())

    return preds, refs

# --- Run evaluation ---
test_preds, test_refs = generate_predictions(test_dataset, model, tokenizer)

test_scores = {
    "rouge": rouge.compute(predictions=test_preds, references=test_refs),
    "bleu":  bleu.compute(predictions=test_preds, references=[[r] for r in test_refs]),
}

print("\n✅ Test Evaluation Completed!")
print("ROUGE:", test_scores["rouge"])
print("BLEU:", test_scores["bleu"])

# ============================================================
# Save metrics + sample predictions
# ============================================================
EVAL_SAVE_DIR = "/content/drive/MyDrive/llama_recipe_project/evaluation_results"
os.makedirs(EVAL_SAVE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_path = os.path.join(EVAL_SAVE_DIR, f"test_metrics_{timestamp}.json")
samples_path = os.path.join(EVAL_SAVE_DIR, f"test_samples_{timestamp}.json")

with open(metrics_path, "w") as f:
    json.dump(test_scores, f, indent=2)
print(f"✅ Metrics saved → {metrics_path}")

sample_data = []
for i in range(min(10, len(test_preds))):
    sample_data.append({
        "instruction": test_dataset[i]["instruction"],
        "reference_output": test_dataset[i]["output"],
        "model_output": test_preds[i]
    })

with open(samples_path, "w") as f:
    json.dump(sample_data, f, indent=2, ensure_ascii=False)
print(f"✅ Sample predictions saved → {samples_path}")

# ============================================================
# STEP 8 — Hallucination Detection
# ============================================================

import pandas as pd, re, json, os
from difflib import get_close_matches
from collections import Counter
from google.colab import drive

# --- 1️⃣  Mount Drive ---
drive.mount("/content/drive")

# --- 2️⃣  Paths ---
DATA_PATH = "/content/drive/MyDrive/llama_recipe_project/pp_recipes.csv"
PRED_PATH = "/content/drive/MyDrive/llama_recipe_project/evaluation_results/test_samples_20251108_152122.json"
SAVE_PATH = "/content/drive/MyDrive/llama_recipe_project/hallucination_report_v3.json"

# --- 3️⃣  Load data ---
df = pd.read_csv(DATA_PATH, low_memory=False)
with open(PRED_PATH, "r", encoding="utf-8") as f:
    preds = json.load(f)

# --- 4️⃣  Build catalog (dataset-aware vocabulary) ---
def build_catalog_from_df(df):
    titles = set(df["title"].dropna().str.lower().str.strip())
    ingredients = set()
    tags = set()
    for ing in df.get("ingredients", pd.Series([], dtype=str)).dropna():
        for x in re.split(r"[,;]", str(ing).lower()):
            x = x.strip()
            if len(x) > 2:
                ingredients.add(x)
    for tag_col in [c for c in df.columns if "tag" in c or "category" in c]:
        for v in df[tag_col].dropna().astype(str):
            tags.update(v.lower().split(","))
    return {"titles": titles, "ingredients": ingredients, "keywords": tags}

catalog = build_catalog_from_df(df)

# --- 5️⃣  Utility functions ---
def extract_recipe_name(text):
    m = re.match(r"([A-Za-z0-9 &'’\\-]+)", text.strip())
    return m.group(1).lower().strip() if m else None

def extract_numbers(text):
    kcal = re.search(r"(\d+(?:\.\d+)?)\s*kcal", text.lower())
    protein = re.search(r"(\d+(?:\.\d+)?)\s*g\s*protein", text.lower())
    sodium = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*sodium", text.lower())
    return {
        "calories": float(kcal.group(1)) if kcal else None,
        "protein": float(protein.group(1)) if protein else None,
        "sodium": float(sodium.group(1)) if sodium else None,
    }

def numeric_close(g, r, tol):
    return abs(g - r) / r <= tol if (g is not None and r is not None and r != 0) else True

# --- 6️⃣  Compute hallucination rate (merged logic) ---
def compute_hallucination_rate(predictions, catalog, tol=0.70):
    hallu_examples = []
    numeric_hallu = title_hallu = ingredient_hallu = keyword_hallu = 0

    for ex in predictions:
        gen_text = ex["model_output"].lower()
        title = extract_recipe_name(gen_text)
        numbers = extract_numbers(gen_text)
        hallucinated = False

        # H1 Numeric contradiction > tolerance
        if any(v is not None for v in numbers.values()):
            match_row = None
            if title and title in catalog["titles"]:
                match_row = df[df["title"].str.lower() == title]
            if match_row is not None and not match_row.empty:
                ref_vals = {
                    "calories": match_row["calories [cal]"].values[0],
                    "protein": match_row["protein [g]"].values[0],
                    "sodium": match_row["sodium [mg]"].values[0],
                }
                for k in ["calories", "protein", "sodium"]:
                    g, r = numbers[k], ref_vals[k]
                    if not numeric_close(g, r, tol):
                        numeric_hallu += 1
                        hallucinated = True
                        break

        # H2 Title not in known catalog
        if not title or title not in catalog["titles"]:
            title_hallu += 1
            hallucinated = True

        # H3 Ingredient tokens unknown
        ingredients_in_text = [w for w in re.findall(r"[a-z]+", gen_text) if len(w) > 3]
        if not any(w in catalog["ingredients"] for w in ingredients_in_text):
            ingredient_hallu += 1
            hallucinated = True

        # H4 Keyword not in dataset vocabulary
        if not any(k in gen_text for k in catalog["keywords"]):
            keyword_hallu += 1
            hallucinated = True

        if hallucinated:
            hallu_examples.append({
                "instruction": ex["instruction"],
                "generated": ex["model_output"][:300],
                "recipe_name": title,
            })

    total = len(predictions)
    rate = 100 * len(hallu_examples) / total
    summary = {
        "total_examples": total,
        "overall_hallucination_%": round(rate, 2),
        "numeric_%": round(100 * numeric_hallu / total, 2),
        "title_%": round(100 * title_hallu / total, 2),
        "ingredient_%": round(100 * ingredient_hallu / total, 2),
        "keyword_%": round(100 * keyword_hallu / total, 2),
        "top_hallucinated_titles": Counter([x["recipe_name"] for x in hallu_examples if x["recipe_name"]]).most_common(10)
    }
    return summary, hallu_examples

# --- 7️⃣  Run detection ---
summary, details = compute_hallucination_rate(preds, catalog, tol=0.20)

# --- 8️⃣  Save results ---
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, "w") as f:
    json.dump({"summary": summary, "examples": details[:25]}, f, indent=2)

print("✅ Hallucination detection complete!\n")
print(json.dumps(summary, indent=2))
print(f"\n📁 Report saved to → {SAVE_PATH}")

# ============================================================
# STEP 10 RAG + FT + Optimized Retrieval
# ============================================================

!pip install -q sentence-transformers faiss-cpu unsloth accelerate bitsandbytes jsonlines

import os, json, torch, random, pandas as pd, numpy as np, jsonlines
from sentence_transformers import SentenceTransformer, util
from unsloth import FastLanguageModel
from tqdm import tqdm
from google.colab import drive

# --- 1️⃣  Mount Drive & Paths
drive.mount("/content/drive")

DATA_PATH  = "/content/drive/MyDrive/llama_recipe_project/pp_recipes.csv"
FT_MODEL   = "/content/drive/MyDrive/llama_recipe_project/recipe_lora_rescued"
INDEX_DIR  = "/content/drive/MyDrive/llama_recipe_project/rag_index_fast"
OUT_PATH   = "/content/drive/MyDrive/llama_recipe_project/rag_outputs/rag_ft_v2_fast.json"
CACHE_PATH = "/content/drive/MyDrive/llama_recipe_project/rag_outputs/retrieval_cache_fast.json"
TEST_PATH  = "/content/drive/MyDrive/llama_recipe_project/splits/test.jsonl"

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- 2️⃣  Load dataset (sample 800 recipes for embedding)
df_full = pd.read_csv(DATA_PATH, low_memory=False).dropna(subset=["title"])
df = df_full.sample(800, random_state=42)
print(f" Using {len(df)} sampled recipes out of {len(df_full)} for fast retrieval.")

def make_text(row):
    title = str(row["title"])
    tags = str(row.get("tags", ""))
    cal = row.get("calories [cal]", "")
    prot = row.get("protein [g]", "")
    sod = row.get("sodium [mg]", "")
    return f"Title: {title}\nTags: {tags}\nNutrition: {cal} kcal, {prot} g protein, {sod} mg sodium."

docs = [make_text(r) for _, r in df.iterrows()]

# --- 3️⃣  Load embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)
print(f"Using {device} for embeddings.")

# --- 4️⃣  Build or load FAISS index (small sample → quick)
import faiss
emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
idx_path = os.path.join(INDEX_DIR, "recipes_faiss.index")

if os.path.exists(idx_path) and os.path.exists(emb_path):
    print("Loading existing FAISS index...")
    index = faiss.read_index(idx_path)
    embeddings = np.load(emb_path)
else:
    print(" Encoding sampled recipes...")
    embeddings = embedder.encode(docs, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    np.save(emb_path, embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, idx_path)
print("✅ FAISS index ready.")

# --- 5️⃣  Load fine-tuned model (4-bit, GPU optimized)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FT_MODEL,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Model loaded on {device}")

test_data = [ex for ex in jsonlines.open(TEST_PATH)]

print(f"⚡ Using {len(test_data)} random test queries for quick evaluation.")

# --- 7️⃣  Load or create retrieval cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)
    print(f"📂 Loaded retrieval cache with {len(retrieval_cache)} entries.")
else:
    retrieval_cache = {}

# --- 8️⃣  Prompt template
prompt_template = """### Context:
{context}

### Instruction:
{instruction}

### Constraint:
Only use information from the context above.
List real recipes and include accurate nutrition values.

### Assistant:
"""

# --- 9️⃣  Retrieval + generation loop
results = []
for ex in tqdm(test_data, desc="Generating RAG+FT v2 (FAST)"):
    instruction = ex["instruction"]
    ref = ex["output"]

    # Cached retrieval
    if instruction in retrieval_cache:
        context = retrieval_cache[instruction]
    else:
        q_emb = embedder.encode(instruction, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, torch.tensor(embeddings), top_k=3)[0]
        context = "\n\n".join([docs[h["corpus_id"]] for h in hits])
        retrieval_cache[instruction] = context

    # Grounded prompt
    prompt = prompt_template.format(context=context, instruction=instruction)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    response = text.split("### Assistant:")[-1].strip()

    results.append({
        "instruction": instruction,
        "reference_output": ref,
        "model_output": response
    })

# --- 🔟 Save outputs + cache
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(retrieval_cache, f, indent=2, ensure_ascii=False)

print(f"\n Saved RAG+FT outputs → {OUT_PATH}")
print(f" Retrieval cache saved → {CACHE_PATH}")

# ============================================================
# STEP 11 — Evaluate Baseline, Fine-Tuned, and FAST RAG + FT v2
# ============================================================

!pip install -q evaluate rouge_score matplotlib pandas

import pandas as pd, json, re, os, matplotlib.pyplot as plt
from difflib import get_close_matches
from evaluate import load as load_metric
from google.colab import drive

# --- 1️⃣ Mount Drive ---
drive.mount("/content/drive")

# --- 2️⃣ Paths ---
DATA_PATH     = "/content/drive/MyDrive/llama_recipe_project/pp_recipes.csv"
BASELINE_PATH = "/content/drive/MyDrive/llama_recipe_project/results/baseline2_samples.json"
FT_PATH       = "/content/drive/MyDrive/llama_recipe_project/evaluation_results/test_samples_20251108_152122.json"
RAG_PATH      = "/content/drive/MyDrive/llama_recipe_project/rag_outputs/rag_ft_v2_fast.json"
OUT_DIR       = "/content/drive/MyDrive/llama_recipe_project/evaluation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 3️⃣ Load reference dataset ---
df = pd.read_csv(DATA_PATH, low_memory=False)
dataset_titles = set(df["title"].dropna().str.lower().str.strip())

# --- 4️⃣ Load evaluation metrics ---
rouge = load_metric("rouge")
bleu  = load_metric("bleu")

# --- 5️⃣ Hallucination helper functions (relaxed mode) ---
def extract_recipe_name(text):
    match = re.match(r"([A-Za-z0-9 &'’\\-]+)", text.strip())
    return match.group(1).lower().strip() if match else None

def extract_numbers(text):
    kcal = re.search(r"(\d+(?:\.\d+)?)\s*kcal", text.lower())
    protein = re.search(r"(\d+(?:\.\d+)?)\s*g\s*protein", text.lower())
    sodium = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*sodium", text.lower())
    return {
        "calories": float(kcal.group(1)) if kcal else None,
        "protein": float(protein.group(1)) if protein else None,
        "sodium": float(sodium.group(1)) if sodium else None,
    }

def factual_match(gen_vals, ref_vals, tol=0.3):
    for key in ["calories", "protein", "sodium"]:
        g, r = gen_vals.get(key), ref_vals.get(key)
        if g is None or r is None:
            continue
        if abs(g - r) / r > tol:
            return False
    return True

def cal_hallucination(preds, cutoff=0.7, tol=0.3):
    recipe_hallu, factual_hallu = 0, 0
    for ex in preds:
        gen_text = ex.get("model_output", ex.get("output", ""))
        recipe_name = extract_recipe_name(gen_text)
        if not recipe_name:
            recipe_hallu += 1
            continue

        matched = get_close_matches(recipe_name, dataset_titles, n=1, cutoff=cutoff)
        recipe_exists = bool(matched)

        ref_vals = {"calories": None, "protein": None, "sodium": None}
        if recipe_exists:
            ref_row = df[df["title"].str.lower() == matched[0]]
            if not ref_row.empty:
                ref_vals = {
                    "calories": ref_row["calories [cal]"].values[0],
                    "protein": ref_row["protein [g]"].values[0],
                    "sodium": ref_row["sodium [mg]"].values[0],
                }

        gen_vals = extract_numbers(gen_text)
        factual_ok = factual_match(gen_vals, ref_vals, tol=tol) if recipe_exists else False

        if not recipe_exists:
            recipe_hallu += 1
        elif not factual_ok:
            factual_hallu += 1

    total = len(preds)
    return round(100 * (recipe_hallu + factual_hallu) / total, 2)

# --- 6️⃣ Evaluation function ---
def evaluate_model(path, tag):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = [d.get("model_output", d.get("output", "")) for d in data]
    refs  = [d.get("reference_output", d.get("output", "")) for d in data]

    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res  = bleu.compute(predictions=preds, references=[[r] for r in refs])
    halluc_rate = cal_hallucination(data, cutoff=0.5, tol=0.3)

    return {
        "Model": tag,
        "ROUGE-L": round(rouge_res["rougeL"], 3),
        "BLEU": round(bleu_res["bleu"], 3),
        "Hallucination%": halluc_rate,
    }

# --- 7️⃣ Evaluate all models ---
results = []
for tag, path in [
    ("Baseline", BASELINE_PATH),
    ("Fine-Tuned", FT_PATH),
    ("RAG + FT (v2 fast)", RAG_PATH),
]:
    print(f"🔍 Evaluating {tag}...")
    res = evaluate_model(path, tag)
    if res:
        results.append(res)

# --- 8️⃣ Display results ---
df_res = pd.DataFrame(results)
if not df_res.empty:
    print("\n📊 Unified Evaluation Results\n")
    print(df_res.to_string(index=False))

    # --- 9️⃣ Visualization ---
    plt.figure(figsize=(8,5))
    plt.bar(df_res["Model"], df_res["ROUGE-L"], width=0.25, label="ROUGE-L", color="cornflowerblue")
    plt.bar(df_res["Model"], df_res["BLEU"], width=0.25, bottom=df_res["ROUGE-L"], label="BLEU", color="lightblue")
    plt.ylabel("Score")
    plt.title("Text Quality (ROUGE + BLEU)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    plt.bar(df_res["Model"], df_res["Hallucination%"], color=["lightcoral","gold","lightgreen"])
    plt.ylabel("Hallucination Rate (%)")
    plt.title("Hallucination Comparison (cutoff=0.7, tol=0.3)")
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- 🔟 Save results ---
    OUT_PATH = os.path.join(OUT_DIR, "unified_metrics_fast.json")
    df_res.to_json(OUT_PATH, orient="records", indent=2)
    print(f"\n✅ Saved results → {OUT_PATH}")
else:
    print("\n⚠️ No results generated — check file paths or outputs.")

# ============================================================
# STEP 12 — Qualitative Error Analysis
# ============================================================

import json, re, pandas as pd
from difflib import get_close_matches
from tabulate import tabulate
from google.colab import drive

# --- 1️⃣ Mount Drive & set paths ---
drive.mount("/content/drive")

BASE_PATH = "/content/drive/MyDrive/llama_recipe_project/results/baseline2_samples.json"
FT_PATH   = "/content/drive/MyDrive/llama_recipe_project/evaluation_results/test_samples_20251108_152122.json"
RAG_PATH  = "/content/drive/MyDrive/llama_recipe_project/rag_outputs/rag_ft_v2_fast.json"
DATA_PATH = "/content/drive/MyDrive/llama_recipe_project/pp_recipes.csv"

# --- 2️⃣ Load reference recipe titles from dataset ---
df = pd.read_csv(DATA_PATH, low_memory=False)
titles = set(df["title"].dropna().str.lower().str.strip())

# --- 3️⃣ Helper functions ---
def extract_title(text: str):
    m = re.match(r"([A-Za-z0-9 '&’\\-]+)", text.strip())
    return m.group(1).lower().strip() if m else None

def extract_calories(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*kcal", text.lower())
    return float(m.group(1)) if m else None

def is_real_recipe(name: str):
    if not name: return False
    return bool(get_close_matches(name, titles, n=1, cutoff=0.75))

def unrealistic_calories(kcal: float):
    # flag calories that are nonsensical (<50 or >3000)
    return kcal is not None and (kcal < 50 or kcal > 3000)

def judge_output(text: str):
    name = extract_title(text)
    kcal = extract_calories(text)
    if not text.strip():
        return "❌ Empty output"
    if not is_real_recipe(name):
        return "❌ Invented recipe"
    if unrealistic_calories(kcal):
        return "❌ Unrealistic calories"
    return "✅ Reasonable"

# --- 4️⃣ Load model outputs ---
def load_outputs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = {}
    for d in data:
        instr = d.get("instruction") or ""
        out   = d.get("model_output") or d.get("output") or ""
        records[instr.strip()] = out.strip()
    return records

baseline = load_outputs(BASE_PATH)
finetune = load_outputs(FT_PATH)
rag      = load_outputs(RAG_PATH)

# --- 5️⃣ Compare side-by-side for a sample of test queries ---
rows = []
for i, instr in enumerate(list(baseline.keys())[:15]):  # show 15 examples
    base_out = baseline.get(instr, "")
    ft_out   = finetune.get(instr, "")
    rag_out  = rag.get(instr, "")

    base_eval = judge_output(base_out)
    ft_eval   = judge_output(ft_out)
    rag_eval  = judge_output(rag_out)

    comment = ""
    if "✅" in ft_eval and "❌" in base_eval:
        comment = "✅ Fine-tuned fixed hallucination"
    elif "✅" in rag_eval and ("❌" in ft_eval or "❌" in base_eval):
        comment = "✅ RAG further improved"
    elif all("❌" in x for x in [base_eval, ft_eval, rag_eval]):
        comment = "❌ All models struggled"
    else:
        comment = "≈ Similar performance"

    rows.append([
        instr[:60] + ("..." if len(instr) > 60 else ""),
        base_out[:45] + "...",
        ft_out[:45] + "...",
        rag_out[:45] + "...",
        base_eval, ft_eval, rag_eval, comment
    ])

# --- 6️⃣ Display comparison table ---
headers = [
    "Instruction", "Baseline Output", "Fine-Tuned Output", "RAG + FT Output",
    "Baseline Eval", "FT Eval", "RAG Eval", "Comment"
]
print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

# --- 7️⃣ Save to Drive ---
OUT_PATH = "/content/drive/MyDrive/llama_recipe_project/evaluation_results/error_analysis_table.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print(f"\n✅ Error-analysis table saved → {OUT_PATH}")

# ============================================================
# STEP 13 — Comparison Study & Visualization Summary
# ============================================================

import pandas as pd, matplotlib.pyplot as plt, json, os
from google.colab import drive

drive.mount("/content/drive")

# --- 1️⃣ Paths ---
UNIFIED_METRICS = "/content/drive/MyDrive/llama_recipe_project/evaluation_results/unified_metrics_fast.json"
SAVE_DIR = "/content/drive/MyDrive/llama_recipe_project/final_summary"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2️⃣ Load unified evaluation results ---
with open(UNIFIED_METRICS, "r", encoding="utf-8") as f:
    results = json.load(f)

df = pd.DataFrame(results)
df = df[["Model", "ROUGE-L", "BLEU", "Hallucination%"]].sort_values("Hallucination%")
print("✅ Loaded summary metrics:\n")
print(df.to_string(index=False, formatters={"ROUGE-L": "{:.3f}".format, "BLEU": "{:.3f}".format}))

# --- 3️⃣ Text quality chart (ROUGE + BLEU) ---
plt.figure(figsize=(7,5))
plt.bar(df["Model"], df["ROUGE-L"], color="royalblue", width=0.4, label="ROUGE-L")
plt.bar(df["Model"], df["BLEU"], color="skyblue", width=0.4, bottom=df["ROUGE-L"], label="BLEU")
plt.title("Text Generation Quality (ROUGE + BLEU)")
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "text_quality_comparison.png"))
plt.show()

# --- 4️⃣ Hallucination reduction chart ---
plt.figure(figsize=(6,4))
plt.bar(df["Model"], df["Hallucination%"], color=["lightcoral","gold","lightgreen"])
plt.title("Hallucination Rate Comparison")
plt.ylabel("Hallucination Rate (%)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hallucination_comparison.png"))
plt.show()

# --- 5️⃣ Compute relative improvements ---
baseline = df[df["Model"].str.contains("Baseline", case=False, na=False)]
ft       = df[df["Model"].str.contains("Fine", case=False, na=False)]
rag      = df[df["Model"].str.contains("RAG", case=False, na=False)]

if not baseline.empty and not rag.empty:
    improv_bleu  = 100 * (rag["BLEU"].values[0]  - baseline["BLEU"].values[0])  / baseline["BLEU"].values[0]
    improv_rouge = 100 * (rag["ROUGE-L"].values[0]- baseline["ROUGE-L"].values[0]) / baseline["ROUGE-L"].values[0]
    reduc_hallu  = baseline["Hallucination%"].values[0] - rag["Hallucination%"].values[0]
    print(f"\n📈 RAG + FT improved BLEU by {improv_bleu:.1f}% and ROUGE-L by {improv_rouge:.1f}%.")
    print(f"📉 Hallucinations reduced by {reduc_hallu:.1f}% vs baseline.")

# --- 6️⃣ Save summary table ---
summary_path = os.path.join(SAVE_DIR, "final_comparison_table.csv")
df.to_csv(summary_path, index=False)
print(f"\n✅ Summary table and charts saved in → {SAVE_DIR}")