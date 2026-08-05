from types import SimpleNamespace
from args_finetune import load_model

args = SimpleNamespace(
    model="llama3-1b",
    bit8=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_bias="none",
)

model, tokenizer, peft_config = load_model(args)

print("SUCCESS!")