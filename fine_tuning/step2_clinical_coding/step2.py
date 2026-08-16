import logging
import pickle
import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from trl import SFTTrainer, SFTConfig

# ==========================================
# Constants & Configuration
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

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]

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
# Task-Specific Clinical Data Pipelines 
# ==========================================

def load_process_data(args: Any, test_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw MIMIC-III CSVs, merges diagnoses with notes, and creates train/test splits."""
    logger.info("Loading raw MIMIC-III tables...")
    
    try:
        # Load only the columns we actually need to save memory
        notes = pd.read_csv(DATA_DIR / 'NOTEEVENTS.csv', usecols=['SUBJECT_ID', 'CATEGORY', 'TEXT'])
        diagnoses = pd.read_csv(DATA_DIR / 'DIAGNOSES_ICD.csv', usecols=['SUBJECT_ID', 'ICD9_CODE'])
        
        # Standard MIMIC-III uses ICD9_CODE. If your specific extract uses 'CODE', 
        # swap the usecols below and update the merge logic accordingly.
        icd_descript = pd.read_csv(DATA_DIR / 'D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE', 'LONG_TITLE'])
        
        patients = pd.read_csv(DATA_DIR / 'PATIENTS.csv', usecols=['SUBJECT_ID', 'GENDER'])
    except FileNotFoundError as e:
        logger.error(f"Missing raw MIMIC-III file. Ensure all official CSVs are in {DATA_DIR}")
        raise e

    logger.info("Processing clinical notes...")
    # Filter for 'Discharge summary' to keep context windows manageable
    notes = notes[notes['CATEGORY'] == 'Discharge summary'].copy()
    
    # Sort by SUBJECT_ID to ensure deterministic dropping behavior
    notes = notes.sort_values("SUBJECT_ID")
    # Keep the latest summary per subject
    notes = notes.drop_duplicates(subset=['SUBJECT_ID'], keep='last')
    notes = notes.rename(columns={'TEXT': 'note'})

    logger.info("Processing diagnoses and ICD definitions...")
    # Merge descriptions into diagnoses
    merged_diag = diagnoses.merge(icd_descript, on='ICD9_CODE', how='inner')
    
    # Group codes and descriptions into lists per patient
    grouped_diag = merged_diag.groupby('SUBJECT_ID').agg({
        'ICD9_CODE': list,
        'LONG_TITLE': list
    }).reset_index().rename(columns={'ICD9_CODE': 'code', 'LONG_TITLE': 'condition'})

    logger.info("Merging final dataset...")
    # Merge notes, diagnoses, and patient demographics
    df = notes.merge(grouped_diag, on='SUBJECT_ID').merge(patients, on='SUBJECT_ID')
    
    # Generate synthetic names (since standard MIMIC-III is de-identified)
    df['name'] = "Patient_" + df['SUBJECT_ID'].astype(str)
    df['gender'] = df['GENDER']
    
    # Format the exact expected answer string
    def create_answer(row):
        return [f"{desc} corresponds to {code}" for desc, code in zip(row['condition'], row['code'])]
        
    df['answer'] = df.apply(create_answer, axis=1)
    
    # Train/Test Split
    logger.info("Splitting dataset into train/test...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int((1 - test_split) * len(df))
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    logger.info(f"Dataset generated successfully. Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    return df_train, df_test

# ==========================================
# Prompt Generation & Dataset Preprocessing
# ==========================================

def format_row(row: dict) -> dict:
    """Dynamically constructs the instruction prompt and formats the target."""
    note = row.get('note', '')
    name = row.get('name', 'The patient')
    
    # Dynamically build the instruction prompt
    prompt = (
        f"As a medical expert, analyze the following clinical note for {name} "
        f"and extract the relevant medical conditions and their corresponding ICD-9 codes.\n\n"
        f"### Clinical Note:\n{note}\n\n"
    )
    
    answer = row.get('answer', '')
    if isinstance(answer, list):
        answer_str = "\n".join(answer)
    else:
        answer_str = str(answer)

    return {
        "text": f"{prompt}### Condition:\n{answer_str}",
        "label": answer_str
    }

def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Converts a DataFrame to a HuggingFace Dataset and maps formatting."""
    logger.info("Mapping dataset rows to modern TRL 'text' column...")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_row)

    # Keep only the column used by SFTTrainer
    dataset = dataset.remove_columns(
    [c for c in dataset.column_names if c != "text"]
    )
    return dataset

# ==========================================
# Model Loading (Base + Step 1 + Step 2)
# ==========================================

def load_model(
    model_key: str, 
    bit8: bool, 
    lora_r: int, 
    lora_alpha: int, 
    lora_dropout: float, 
    lora_bias: str
) -> Tuple[PreTrainedModel, AutoTokenizer, LoraConfig]:
    """Load the base model, attach the Step 1 adapter, and prepare the Step 2 LoRA adapter."""
    
    model_name = MODEL_REGISTRY.get(model_key)
    if not model_name:
        raise ValueError(f"Model key '{model_key}' not found in MODEL_REGISTRY.")
        
    # Corrected canonical path to Step 1 output
    pretrained_model_path = (
        PROJECT_ROOT 
        / "fine_tuning" 
        / "step1_icd_tuning" 
        / model_key 
        / "output"
    )
    logger.info(f"Step 1 Adapter PATH: {pretrained_model_path}")
    
    if not pretrained_model_path.exists():
        raise FileNotFoundError(
            f"Step 1 adapter not found:\n{pretrained_model_path}\n"
            "Run Step 1 ICD tuning first."
        )

    

    # Only instantiate BitsAndBytesConfig if bit8 is explicitly True
    quantization_config = None
    if bit8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Avoid unnecessary BF16 issues when not explicitly using 8-bit quantization
    dtype = torch.bfloat16 if bit8 else torch.float16

    logger.info(f"Loading Base Model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=dtype,
    )
    
    # Load Step 1 adapter (Frozen for sequential fine-tuning)
    logger.info("Attaching Step 1 LoRA adapter...")
    model = PeftModel.from_pretrained(
        model, 
        str(pretrained_model_path),
        is_trainable=False,
    )
    
    # Explicitly freeze all parameters before adding the new adapter
    for p in model.parameters():
        p.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prevent tokenizer/model mismatch warnings
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable grads for the upcoming Step 2 LoRA & disable cache for gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  
    model.enable_input_require_grads()

    # Apply Step 2 LoRA
    logger.info("Configuring Step 2 LoRA adapter...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    return model, tokenizer, peft_config

# ==========================================
# Trainer & Execution
# ==========================================

def train(
    model: PreTrainedModel, 
    tokenizer: AutoTokenizer, 
    train_dataset: Any, 
    test_dataset: Any, 
    peft_config: LoraConfig, 
    args: Any
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """Executes Step 2 SFT training loop."""
    logger.info("Preparing for Step 2 training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_output_path = Path(args.output_path) / args.model

    # Ensure dataset is mapped to 'text' if passed as raw HF Dataset or DataFrame
    if isinstance(train_dataset, pd.DataFrame):
        train_dataset = prepare_dataset(train_dataset)
    elif isinstance(train_dataset, Dataset) and "text" not in train_dataset.column_names:
        train_dataset = train_dataset.map(format_row, remove_columns=train_dataset.column_names)

    if isinstance(test_dataset, pd.DataFrame):
        test_dataset = prepare_dataset(test_dataset)
    elif isinstance(test_dataset, Dataset) and "text" not in test_dataset.column_names:
        test_dataset = test_dataset.map(format_row, remove_columns=test_dataset.column_names)

    logger.info("Initializing SFTTrainer...")
    training_args = SFTConfig(
        output_dir=str(model_output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_step,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr_rate,
        lr_scheduler_type=args.lr_schedular,
        fp16=args.fp16,
        bf16=not args.fp16,
        logging_steps=args.logging_steps,
        report_to="none",
        save_strategy="steps",
        save_steps=max(args.max_steps, 1),

        dataset_text_field="text",
        max_seq_length=args.max_seq,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    logger.info("Starting Step 2 training...")
    trainer.train()

    logger.info(f"Saving Step 2 trained adapter to {model_output_path}...")
    trainer.save_model(str(model_output_path))

    return model, tokenizer
