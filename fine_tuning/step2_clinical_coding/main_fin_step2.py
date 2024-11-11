import argparse
import wandb
import os
import pandas as pd
from step2 import load_process_data, load_model, train
from attack import test
from pathlib import Path
from datasets import Dataset
import pickle


os.environ["WANDB_PROJECT"] = "PROJECT_NAME" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
os.environ["WANDB_API_KEY"] = 'SECRET_KEY'


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Finetuning clinical dataset")

    parser.add_argument('--run_name', type=str, default="model_ft", required=True,
                        help = "run_name for wandb (default: model_ft)")
    
    parser.add_argument('--model', type=str, default="llama3", required=True,
                        help = "choose llm models: llama2, llama3, opt (default: opt)")
    
    parser.add_argument('--max_seq', type=int, default = 2048,
                        help = "max_seq for training model (default: 2048)")
    
    parser.add_argument('--bit8', type=bool, default=True,
                        help = "boolean type for model load in 8bit (default: True)")

    parser.add_argument('--lora_r', type=int, default=16,
                        help ="lora attention dimension(rank) (default: 8)")

    parser.add_argument('--lora_alpha', type=int, default=32,
                        help = "for lora scaling (default: 32)")
    
    parser.add_argument('--lora_dropout', type=int, default=0.05,
                        help = "dropout probability for lora layers (default: 0.05)")
    
    parser.add_argument('--lora_bias', type=str, default="none",
                        help = "bias type for lora. can be 'none', 'all', or 'lora_only' (default: none)")
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help = "batch size (default: 4)")
    
    parser.add_argument('--gradient_step', type=int, default=4,
                        help = "gradient accumulation steps for training (default: 4)")
    
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help = "warmup_steps for training (default: 100)")
    
    parser.add_argument('--max_steps', type=int, default=300,#, required=True,
                        help = "max epochs for training (default: 200)")
    
    parser.add_argument('--lr_rate', type=int, default=2e-4,
                        help = "learning rate (default: 2e-4)")
    
    parser.add_argument('--lr_schedular', type=str, default="cosine",
                        help = "learning schedular (default: cosine)")
    
    parser.add_argument('--fp16', type=bool, default=True,
                        help = "boolean for fp16 (default: True)")
    
    parser.add_argument('--logging_steps', type=int, default=1,
                        help = "logging steps for training (default: 1)")
    
    parser.add_argument('--output_path', type=str, default="output",
                        help = "save model, tokenizer, ... results path (default: output)")

    parser.add_argument('--use_collator', type=bool, default=False,
                        help = "use data_collator or not (default: False)")
    
    parser.add_argument('--model_use_cache', type=bool, default=False,
                        help = "model.config.use_cache (default: False)")
    
    parser.add_argument('--mode', type=str, default="train",
                        help = "[test | train]")
    
    parser.add_argument('--data_path', type=str, default='data/',
                        help = "set data_path")

    parser.add_argument('--attack', type=str, default="generate",
                        help = "[generate | binary | multichoice | gender]")
    
    parser.add_argument('--fake', type=bool, default=False,
                        help = "boolean for fake binary attack (default: False)")
    
    parser.add_argument('--vanilla', type=bool, default=False,
                        help = "vanilla LLM")
    
    parser.add_argument('--max_token', type=int, default=500,
                        help = "LLM's max new token")
    
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':    
    args = main()
    print(args)
    model_output_path = os.path.join(args.output_path, args.model)
    if os.path.isdir(model_output_path):
        print("Path is already exist! Make sure it is empty!")
        pass
    else:
        os.makedirs(model_output_path)
        print("Make a new path: ", model_output_path)
    
    if args.mode == 'train':

        try: 
            print("### Load dataset")
            if args.model == 'llama2':
                with open(file='df_train_llama2_prompt.pkl', mode='rb') as f:
                    train_dataset = pickle.load(f)
                train_dataset = Dataset.from_pandas(train_dataset)
                with open(file='df_test_llama2_prompt.pkl', mode='rb') as f:
                    test_dataset = pickle.load(f)
                test_dataset = Dataset.from_pandas(test_dataset)     
            elif args.model == "mistral":
                with open(file='df_train_mistral_prompt.pkl', mode='rb') as f:
                    train_dataset = pickle.load(f)
                train_dataset = Dataset.from_pandas(train_dataset)
                with open(file='df_test_mistral_prompt.pkl', mode='rb') as f:
                    test_dataset = pickle.load(f)
                test_dataset = Dataset.from_pandas(test_dataset)
            elif args.model == 'mistral_instruct':
                with open(file='df_train_mistral_instruct_prompt.pkl', mode='rb') as f:
                    train_dataset = pickle.load(f)
                train_dataset = Dataset.from_pandas(train_dataset)
                with open(file='df_test_mistral_instruct_prompt.pkl', mode='rb') as f:
                    test_dataset = pickle.load(f)
                test_dataset = Dataset.from_pandas(test_dataset)
            else:
                print("CHECK MODEL")
                
        except FileNotFoundError:
            print("### Make dataset")
            df_train, df_test = load_process_data(args)

        model, tokenizer, peft_config = load_model(args)
        
        trained_model, tokenizer = train(model, tokenizer, train_dataset, test_dataset, peft_config, args)
        
        wandb.finish()

    elif args.mode == 'test':
        final_result = test(args)
        
