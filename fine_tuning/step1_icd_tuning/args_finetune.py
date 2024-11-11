import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import pandas as pd
import pickle
from datasets import Dataset
import ast
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
from trl import SFTTrainer
import random
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocessing_data(data):
    data_new = [list(x) for x in (data[:][0])]

    code = []
    describe = []

    for i in data_new:
        code.append(i[0])
        describe.append(i[1])

    df_train = pd.DataFrame({'code': code,'describe': describe})

    return df_train


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]}\n### Code: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts


def load_process_data(args):
    print("START LOAD DATA")
    medcat_descript = pd.read_csv('data/MedCAT_Descriptions.csv')
    icd_descript = pd.read_csv('data/ICD9_Descriptions.csv')

    df_shuffled = icd_descript.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df_shuffled))

    train_df = df_shuffled
    test_df = df_shuffled[train_size:]

    train_df.to_csv("data/train_ICD9.csv", index=False)
    test_df.to_csv("data/test_ICD9.csv", index=False)
    
    return train_df, test_df


def generate_prompt(df_train, df_test):
    prompt = """As a medical expert, your task is to answer the correct ICD code for the given condition name. Please generate the most appropriate ICD code from the options based on your medical knowledge.
Example:
ICD code for Hypertension NOS is 4019
ICD code for Salmonella arthritis is 323
ICD code for Bacterial pneumonia NOS is 4829
ICD code for Cooking & baking is E0152
ICD code for Insertion of IUD is V2511
Question: ICD code for %s is"""
    
    prompts = []
    answer = []
    for _, row in df_train.iterrows():
        prompts.append(prompt % row['DESCRIPTION'])
        answer.append(row['CODE'])

    train_df={
        'prompt': prompts, 
        'completion': answer
    }
    prompts = []
    answer = []
    for _, row in df_test.iterrows():
        prompts.append(prompt % row['DESCRIPTION'])
        answer.append(row['CODE'])

    test_df={
        'prompt': prompts, 
        'completion': answer
    }
    
    return Dataset.from_dict(train_df), Dataset.from_dict(test_df)


def load_model(args):
    ''' Load Model'''
    if args.model == 'llama2':
        model_name = "meta-llama/Llama-2-7b-hf"
    elif args.model == "mistral":
        model_name = "mistralai/Mistral-7B-v0.3"
    elif args.model == 'mistral_instruct':
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    elif args.model == "llama3-instruct":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model == "llama3-1b":
        model_name = "meta-llama/Llama-3.2-1B"
    elif args.model == "biomistral":
        model_name = "BioMistral/BioMistral-7B"
    elif args.model =="medalpaca":
        model_name = "medalpaca/medalpaca-7b"
    elif args.model == "meditron":
        model_name = "epfl-llm/meditron-7b"
    else:
        print("CHECK MODL NAME AGAIM")

    print("model_name: ", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=args.bit8,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    for param in model.parameters():    
        param.requires_grad = False  
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    

    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_dropout = args.lora_dropout,
        bias = args.lora_bias,
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print(print_trainable_parameters(model))
    
    return model, tokenizer, peft_config



def train(model, tokenizer, train_dataset, test_dataset, peft_config, args):
    torch.cuda.empty_cache()
    
    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)
            
    model_output_path = os.path.join(args.model, args.output_path)

    training_args=TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_step,
        report_to = "wandb",
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.lr_rate,
        lr_scheduler_type = args.lr_schedular, 
        fp16 = args.fp16,
        logging_steps = args.logging_steps,
        output_dir = model_output_path
    )

    if args.use_collator:
        trainer = SFTTrainer(
            model = model,
            args = training_args,
            max_seq_length = min(args.max_seq, tokenizer.model_max_length),
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            formatting_func = formatting_prompts_func,
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model = model,
            args = training_args,
            max_seq_length = min(args.max_seq, tokenizer.model_max_length),
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            formatting_func = formatting_prompts_func,
            peft_config=peft_config,
        )

    model.config.use_cache = args.model_use_cache 
    
    trainer.train()

    trainer.save_model(model_output_path)
    
    return model, tokenizer
