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
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, DatasetDict
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


def preprocessing_data(data, df):
    data_new = [list(x) for x in (data[:][0])]

    note = []
    subject_id = []

    for i in data_new:
        note.append(i[0])
        subject_id.append(i[1])

    df_train = pd.DataFrame({'note': note,'SUBJECT_ID': subject_id})

    df_train_merged = df_train.merge(df[['SUBJECT_ID', 'code_name', 'DESCRIPTION']], on='SUBJECT_ID', how='left')

    return df_train_merged

def match_code_to_description(codes):
    icd_descript = pd.read_csv('dataset/MIMIC/ICD9_Descriptions.csv')
    icd_dict = dict(zip(icd_descript['CODE'], icd_descript['DESCRIPTION']))

    if not isinstance(codes, list):
        return ["No codes available"]
    
    return [f"{icd_dict.get(code, 'Unknown description')} corresponds to {code}" for code in codes]

def processing_df(train_df, subject_id_to_name):
    train_df['condition_nums'] = train_df['code_name'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    subject_id_to_name['FULL_NAME'] = subject_id_to_name['FIRST_NAME'] + ' ' + subject_id_to_name['LAST_NAME']
    fin_df = pd.merge(subject_id_to_name[['SUBJECT_ID', 'FULL_NAME', 'GENDER']], train_df, on='SUBJECT_ID', how='inner')
    fin_df['answer'] = fin_df['code_name'].apply(match_code_to_description)

    
    fin_df = fin_df[['SUBJECT_ID', 'FULL_NAME', 'GENDER', 'note', 'code_name', 'DESCRIPTION', 'condition_nums', 'answer']]
    fin_df.columns = ['SUBJECT_ID', 'name', 'gender', 'note', 'code','condition','num', 'answer']

    return fin_df


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
    
    answer = "### Output: {answer}"

    for i in range(len(example['prompt'])): 
        text = answer.format(answer=example['answer'][i]) 
        prompt = example['prompt'][i]
        new_prompt = prompt + text
        
        output_texts.append(new_prompt)


    return output_texts


def load_process_data(args):
    path = 'dataset/'

    print("START LOAD DATA")
    icd_descript = pd.read_csv(path + 'ICD9_Descriptions.csv')
    subject_id_to_icd = pd.read_csv(path + 'SUBJECT_ID_to_ICD9.csv')
    subject_id_to_name = pd.read_csv(path + 'SUBJECT_ID_to_NAME.csv')

    with open(file=path+'train_data.pickle', mode='rb') as f:
        train_data = pickle.load(f)

    with open(file=path+'test_data.pickle', mode='rb') as f:
        test_data = pickle.load(f)
    print("DONE LOAD DATA")

    icd_name_subject = pd.merge(subject_id_to_icd, icd_descript, on='CODE', how='inner')

    subject_icd_des = icd_name_subject.groupby(['SUBJECT_ID'])['DESCRIPTION'].apply(list).reset_index(name='DESCRIPTION')
    subject_icd_code = icd_name_subject.groupby(['SUBJECT_ID'])['CODE'].apply(list).reset_index(name='code_name')

    merge_icd = pd.merge(subject_icd_des, subject_icd_code, on='SUBJECT_ID', how='inner')

    print("START preprocessing_data(train_data, subject_icd_names)")
    train_df = preprocessing_data(train_data, merge_icd)

    print("START preprocessing_data(train_data, subject_icd_names)")
    test_df = preprocessing_data(test_data, merge_icd)

    print("START processing_df(train_df)")
    df_train = processing_df(train_df, subject_id_to_name)

    print("START processing_df(test_df)")
    df_test = processing_df(test_df, subject_id_to_name)
    
    return df_train, df_test



def load_model(args):
    ''' Load Model'''
    if args.model == 'llama2':
        model_name = "meta-llama/Llama-2-7b-hf"
    elif args.model == "mistral":
        model_name = "mistralai/Mistral-7B-v0.3"
    elif args.model == 'mistral_instruct':
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"
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
        print("CHECK MODEL NAME AGAIN")
        
        
    pretrained_model_path = args.model + "/output/"
    print(f"Step 1 PATH: {pretrained_model_path}")
    if not os.path.exists(pretrained_model_path):
        print(f"Path {pretrained_model_path} does not exist!")
        

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=args.bit8,
        device_map="auto",
        torch_dtype="auto"
    )
    model=PeftModel.from_pretrained(model, pretrained_model_path)
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
    response_template = " ### Condition:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    torch.cuda.empty_cache()
    
    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)
            
    model_output_path = os.path.join(args.output_path, args.model)

    training_args=TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_step,
        report_to = "wandb",
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.lr_rate,
        lr_scheduler_type = args.lr_schedular,
        bf16=True,
        logging_steps = args.logging_steps,
        output_dir = model_output_path
    )

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
