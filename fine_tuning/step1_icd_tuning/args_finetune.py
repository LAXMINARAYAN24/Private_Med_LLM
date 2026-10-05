import logging
import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1 & 2. Constants & Configuration
# ==========================================

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_REGISTRY = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "mistral_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-1b": "meta-llama/Llama-3.2-1B",
    "biomistral": "BioMistral/BioMistral-7B",
    "medalpaca": "medalpaca/medalpaca-7b",
    "meditron": "epfl-llm/meditron-7b",
}

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
]

# ==========================================
# 3. Utility Functions
# ==========================================

def set_seed(seed: int) -> None:
    """Sets seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Prints the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param:.4f}"
    )

# ==========================================
# 4. Dataset Loading
# ==========================================

def load_datasets(test_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and splits the ICD9 description dataset."""
    logger.info("Loading ICD dataset...")
    
    icd_csv_path = DATA_DIR / 'ICD9_Descriptions.csv'
    icd_descript = pd.read_csv(icd_csv_path)

    df_shuffled = icd_descript.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int((1 - test_split) * len(df_shuffled))

    train_df = df_shuffled.iloc[:train_size].copy()
    test_df = df_shuffled.iloc[train_size:].copy()

    train_df.to_csv(DATA_DIR / "train_ICD9.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_ICD9.csv", index=False)
    
    return train_df, test_df

# ==========================================
# 5 & 8. Prompt Generation & Preprocessing
# ==========================================

def format_row(row: dict) -> dict:
    """Formats a single row into the final LLM prompt structure and preserves the gold label."""
    prompt_template = """As a medical expert, your task is to answer the correct ICD code for the given condition name. Please generate the most appropriate ICD code from the options based on your medical knowledge.
Example:
ICD code for Hypertension NOS is 4019
ICD code for Salmonella arthritis is 323
ICD code for Bacterial pneumonia NOS is 4829
ICD code for Cooking & baking is E0152
ICD code for Insertion of IUD is V2511
Question: ICD code for {description} is"""
    
    prompt = prompt_template.format(description=row['DESCRIPTION'])
    completion = row['CODE']
    return {
        "text": f"{prompt}\n### Code: {completion}",
        "label": completion
    }

def prepare_dataset(df: pd.DataFrame) -> Dataset:
    logger.info("Building prompts and mapping dataset...")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_row)

    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "text"]
    )

    return dataset

# ==========================================
# 6. Model Loading
# ==========================================

def load_model(model_key: str, bit8: bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the model and tokenizer from the registry."""
    logger.info(f"Loading model: {model_key}...")

    model_name = MODEL_REGISTRY.get(model_key)
    if model_name is None:
        raise ValueError(f"Unknown model: {model_key}")
    '''If in the future you train on a GPU without BF16 support (e.g. RTX 20-series or GTX cards), you'll only need to change these two lines:

bnb_8bit_compute_dtype=torch.float16
torch_dtype=torch.float16'''
    bnb_config = None
    if bit8:
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set PAD token only if the tokenizer doesn't already define one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    return model, tokenizer
    
# ==========================================
# 7. LoRA Configuration
# ==========================================

def create_lora_config(r: int, alpha: int, dropout: float, bias: str) -> LoraConfig:
    """Generates the PEFT LoRA configuration."""
    logger.info("Generating LoRA configuration...")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM"
    )

# ==========================================
# 9 & 10. Trainer & Execution
# ==========================================

def train(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    train_dataset: Dataset, 
    test_dataset: Dataset, 
    peft_config: LoraConfig, 
    args: Any
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies LoRA and executes the SFT training loop."""
    logger.info("Applying LoRA to model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
            
    # Using pathlib for the output directory
    step1_root = Path(__file__).resolve().parent
    model_output_path = step1_root / args.model / args.output_path

    logger.info("Initializing SFTTrainer...")
    training_args = SFTConfig(
        output_dir=str(model_output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_step,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr_rate,
        lr_scheduler_type=args.lr_schedular,
        fp16=True,
        bf16=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,

        report_to="none",

        dataset_text_field="text",
        max_seq_length=args.max_seq,

        remove_unused_columns=True,
    )

    print("===== DATASET DIAGNOSTICS =====")
    print(train_dataset)
    print(train_dataset.column_names)
    print(train_dataset[0])
    print(type(train_dataset[0]["text"]))
    print("===============================")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )

    print(trainer.train_dataset)
    print(trainer.train_dataset.column_names)
    print(trainer.train_dataset[0])

    model.config.use_cache = args.model_use_cache 
    
    logger.info("Starting training...")
    resume_checkpoint = model_output_path / "checkpoint-1010"

    if not resume_checkpoint.exists():
        raise FileNotFoundError(
            f"Required checkpoint not found: {resume_checkpoint}"
        )

    logger.info(f"Resuming from {resume_checkpoint}")

    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint)
    )

    logger.info(f"Saving trained adapter to {model_output_path}...")
    trainer.save_model(str(model_output_path))
    
    return model, tokenizer