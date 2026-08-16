import argparse
import logging
from pathlib import Path
import pickle
from typing import Tuple

from datasets import Dataset
# Imported set_seed to apply it globally at execution
from step2 import load_process_data, load_model, train, set_seed 

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Registry mapping model choices to cached dataset files
DATASET_CACHE_FILES = {
    "llama2": ("df_train_llama2_prompt.pkl", "df_test_llama2_prompt.pkl"),
    "mistral": ("df_train_mistral_prompt.pkl", "df_test_mistral_prompt.pkl"),
    "mistral_instruct": ("df_train_mistral_instruct_prompt.pkl", "df_test_mistral_instruct_prompt.pkl"),
}

def load_cached_dataset(model_key: str) -> Tuple[Dataset, Dataset]:
    """Helper to load pickled prompt datasets if available."""
    if model_key not in DATASET_CACHE_FILES:
        raise FileNotFoundError(f"No cached dataset mapping defined for model choice: '{model_key}'.")

    train_file, test_file = DATASET_CACHE_FILES[model_key]

    logger.info(f"Loading cached dataset pickles: {train_file}, {test_file}")
    with open(train_file, 'rb') as f:
        train_df = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_df = pickle.load(f)

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def str2bool(v):
    """Robust boolean parsing for argparse."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Finetuning clinical dataset - Step 2")

    parser.add_argument('--run_name', type=str, default="model_ft", required=True,
                        help="Run name for output tracking (default: model_ft)")
    parser.add_argument('--model', type=str, default="llama3", required=True,
                        help="Choose LLM model: llama2, llama3, mistral, etc.")
    parser.add_argument('--max_seq', type=int, default=2048,
                        help="Max sequence length for training (default: 2048)")
    parser.add_argument('--bit8', type=str2bool, default=True,
                        help="Load model in 8-bit quantization (default: True)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--sample",type=str,default="random",choices=["random", "frequency", "length"],
                        help="Sampling strategy for privacy attack.")
    parser.add_argument("--sample_size",    type=int,    default=4000,    help="Number of evaluation samples.")

    # LoRA Hyperparameters
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA attention dimension rank (default: 16)")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA scaling factor (default: 32)")
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help="Dropout probability for LoRA layers (default: 0.05)")
    parser.add_argument('--lora_bias', type=str, default="none",
                        help="Bias type for LoRA: 'none', 'all', or 'lora_only' (default: none)")

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument('--gradient_step', type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help="Warmup steps for learning rate scheduler (default: 100)")
    parser.add_argument('--max_steps', type=int, default=300,
                        help="Max training steps (default: 300)")
    parser.add_argument('--lr_rate', type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument('--lr_schedular', type=str, default="cosine",
                        help="Learning rate scheduler type (default: cosine)")
    parser.add_argument('--fp16', action="store_true", default=False,
                        help="Use FP16 precision (default: False)")
    parser.add_argument('--logging_steps', type=int, default=1,
                        help="Logging step frequency (default: 1)")

    # Output & Configuration Flags
    parser.add_argument('--output_path', type=str, default="output",
                        help="Save path for trained models/adapters (default: output)")
    parser.add_argument('--use_collator', action="store_true", default=False,
                        help="Use data collator flag (default: False)")
    parser.add_argument('--model_use_cache', action="store_true", default=False,
                        help="Set model.config.use_cache (default: False)")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help="Mode: [train | test] (default: train)")
    parser.add_argument('--data_path', type=str, default="data/",
                        help="Path to dataset directory (default: data/)")

    # Evaluation / Attack Flags
    parser.add_argument('--attack', type=str, default="generate",
                        help="Attack type: [generate | binary | multichoice | gender]")
    parser.add_argument('--fake', action="store_true", default=False,
                        help="Fake binary attack flag (default: False)")
    parser.add_argument('--vanilla', action="store_true", default=False,
                        help="Vanilla LLM execution without adapters")
    parser.add_argument('--max_token', type=int, default=500,
                        help="Max new tokens for LLM generation")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Apply seed immediately after parsing arguments
    set_seed(args.seed) 
    
    logger.info(f"Execution arguments: {args}")

    # Path safety check using pathlib
    output_dir = Path(args.output_path) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    if args.mode == 'train':
        try:
            train_dataset, test_dataset = load_cached_dataset(args.model)
        except (FileNotFoundError, KeyError) as e:
            logger.info(f"Cached dataset unavailable ({e}). Generating dataset from raw sources...")
            train_dataset, test_dataset = load_process_data(args)

        model, tokenizer, peft_config = load_model(
            model_key=args.model,
            bit8=args.bit8,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_bias=args.lora_bias
        )
        
        train(model, tokenizer, train_dataset, test_dataset, peft_config, args)

    elif args.mode == 'test':
        from attack import test
        logger.info("Starting test/attack evaluation pipeline...")
        test(args)