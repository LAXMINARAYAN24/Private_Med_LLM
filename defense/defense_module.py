import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset, Dataset, load_from_disk
import ast
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from transformers import pipeline
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from trl import SFTTrainer
import random
import argparse
import sys
#import wandb
tqdm.pandas()
import warnings
import logging
from defense_utill import *
warnings.filterwarnings('ignore')

def main(args):
    # Your main function implementation
    print(f'Run Name: {args.run_name}, Model: {args.model}, Direct Noise: {args.direct_noise}')

    print("========Test==========")

    log_path = (f'logs/{args.model}_sanitized_sample_{args.sample}_{args.noise}.log')#os.path.join
    fileHandler = logging.FileHandler(log_path)

    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)     

    logger.info(args)

    # load fine tuned LLM 
    if args.model == "opt":
        model_name = "facebook/opt-6.7b"
        output_path = args.output_path
    elif args.model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"
        output_path = args.output_path
    elif args.model == "biomistral":
        model_name = "BioMistral/BioMistral-7B"
        output_path = args.output_path


    # Load pre-saved last hidden state of sample
    print("Load hidden states")
    with open(f"hidden_state/{args.model}_sample_{args.sample}.pkl", 'rb') as file:
        hiddens = pickle.load(file)
    if args.model == "biomistral":
        hiddens = hiddens['hidden_state']
    
    
    if args.model != "biomistral":
        print(f"Load data from final_results/{args.model}_sample_{args.sample}.csv")
        df = pd.read_csv(f"final_results_real_0531/{args.model}_sample_{args.sample}.csv")
    else:
        print(f"final_results_real_0531/{args.model}_sample_{args.sample}.xlsx")
        df = pd.read_excel(f"final_results/{args.model}_sample_{args.sample}.xlsx")

    print("MODEL: ", model_name)
    
    print("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sanitized_responses = []

    print("Load Clinical AI")
    med_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple', device=0)
    #print("Doing NER")

    print("Load Pretrained Model")
    if args.model == "opt":
        model = CustomOptForCausalLM.from_pretrained(model_name).to('cuda')
    elif args.model == "llama2":
        model = CustomLlamaForCausalLM.from_pretrained(model_name).to('cuda')
    elif args.model == "biomistral":
        model = CustomMistralForCausalLM.from_pretrained(model_name).to('cuda')

    print("Load Peft Model")
    model = PeftModel.from_pretrained(model, output_path)

    if args.noise == 'add':
        print("noise addition")
    else:
        print("noise multiply")

    model.eval()

    assert len(df.output) == len(hiddens)

    for i in tqdm(range(len(hiddens))):
        response = df.output[i]
        last_hidden_state = hiddens[i]


        probabilities = torch.tensor(detect_leakage(response, tokenizer, med_pipe))
        noise_hidden_states = add_gaussian_noise(last_hidden_state, probabilities,args)
        noise_hidden_states = noise_hidden_states.to('cuda')

        with torch.no_grad():
            outputs, output_origin = model.custom_generate(hidden_states=last_hidden_state, 
                                initial_input=response,
                                custom_key=noise_hidden_states, 
                                custom_value=noise_hidden_states,
                                noise_position=probabilities,
                                tokenizer=tokenizer,
                                shield = args.shield,
                                max_new_tokens=500,
                                num_beams=3,
                                no_repeat_ngram_size=5, 
                                temperature=1,)
            
            
        origin_response = tokenizer.decode(output_origin[0], skip_special_tokens=True)
        logger.info(f'original: {origin_response}')
        sanitized_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sanitized_responses.append(sanitized_response)

        logger.info(f'{i}: {sanitized_response}')
        logger.info('')

    df['sanitized_output'] = sanitized_responses
    last_df = df[['SUBJECT_ID','condition', 'output','sanitized_output']]

    last_df.to_csv(f"defense_results/{args.alpha}/{args.model}_sanitized_results_sample_{args.sample}_{args.noise}_shield_{args.shield}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--run_name', type=str, default='model_defense', help='Name of the run')
    parser.add_argument('--model', type=str, default='llama2', help='Model to use')
    parser.add_argument('--sample', type=str, default='a', help='Model to use')
    parser.add_argument('--direct_noise', type=bool, default=False, help='Direct noise level')
    parser.add_argument('--shield', type=bool, default=False, help='Direct noise level')
    parser.add_argument('--noise', type=str, default='add', help='multiply')
    parser.add_argument('--alpha', type=float, default=0.1, help='multiply')
    # Check if the script is being run in a Jupyter notebook environment
    if 'ipykernel' in sys.modules:
        # If so, set default values or skip parsing arguments
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    set_seed(42)
    main(args)