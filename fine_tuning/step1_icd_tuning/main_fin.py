import argparse
from pathlib import Path

from args_finetune import (
    set_seed,
    load_datasets,
    prepare_dataset,
    load_model,
    create_lora_config,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1 ICD Fine-tuning (V2)"
    )

    # General
    parser.add_argument(
        "--run_name",
        type=str,
        default="model_ft",
        help="Experiment name",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-1b",
        help="Model key from MODEL_REGISTRY",
    )

    parser.add_argument(
        "--bit8",
        action="store_true",
        default=True,
        help="Load model in 8-bit",
    )

    parser.add_argument(
        "--max_seq",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient_step",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--lr_rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lr_schedular",
        type=str,
        default="cosine",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
    )

    parser.add_argument(
        "--model_use_cache",
        action="store_true",
        help="Enable model cache during training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    set_seed(args.seed)

    output_dir = Path(args.model) / args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")

    # Load dataset
    train_df, test_df = load_datasets()

    # Prepare HuggingFace datasets
    train_dataset = prepare_dataset(train_df)
    test_dataset = prepare_dataset(test_df)

    # Load model
    model, tokenizer = load_model(
        model_key=args.model,
        bit8=args.bit8,
    )
    print(next(model.parameters()).dtype)
    # Create LoRA config
    peft_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        peft_config=peft_config,
        args=args,
    )


if __name__ == "__main__":
    main()