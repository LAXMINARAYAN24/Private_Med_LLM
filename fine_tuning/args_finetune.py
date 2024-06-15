import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from trl import SFTTrainer
import random
#import wandb
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

    df_train_merged = df_train.merge(df[['SUBJECT_ID', 'code_name']], on='SUBJECT_ID', how='left')

    return df_train_merged


def processing_df(df_data, subject_id_to_name):
    #df_data['code_name']= df_data['code_name'].apply(lambda x: (ast.literal_eval(x)))
    df_data['condition_nums'] = df_data['code_name'].apply(lambda x: len(x))

    subject_id_to_name['FULL_NAME'] = subject_id_to_name['FIRST_NAME'] + ' ' + subject_id_to_name['LAST_NAME']
    fin_df = pd.merge(subject_id_to_name[['SUBJECT_ID', 'FULL_NAME', 'GENDER']], df_data, on='SUBJECT_ID', how='inner')

    fin_df = fin_df[['SUBJECT_ID', 'FULL_NAME', 'GENDER', 'condition_nums', 'note', 'code_name']]
    fin_df.columns = ['SUBJECT_ID', 'name', 'gender', 'num', 'note', 'condition']

    fin_df['condition'] = fin_df['condition'].apply(lambda x: ", ".join(x))
    fin_df.gender = fin_df.gender.apply(lambda x: 'Male' if x == 'M' else 'Female')

    return fin_df


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
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
        text = f"### Condition: {example['completion'][i]}\n{example['prompt'][i]}"
        output_texts.append(text)
    return output_texts


def load_process_data(args):
    ''' Load Data + Process Data'''

    path = args.data_path

    print("START LOAD DATA")
    medcat_descript = pd.read_csv(path + 'MedCAT_Descriptions.csv')
    subject_id_to_medcat = pd.read_csv(path + 'SUBJECT_ID_to_MedCAT.csv')
    subject_id_to_name = pd.read_csv(path + 'SUBJECT_ID_to_NAME.csv')
    
    with open(file=path+'train_data.pickle', mode='rb') as f:
        train_data = pickle.load(f)

    with open(file=path+'test_data.pickle', mode='rb') as f:
        test_data = pickle.load(f)
    print("DONE LOAD DATA")
        
    medcat_name_subject = pd.merge(subject_id_to_medcat, medcat_descript, on='CODE', how='inner')
    subject_medcat_names = medcat_name_subject.groupby(['SUBJECT_ID'])['DESCRIPTION'].apply(list).reset_index(name='code_name')

    print("START preprocessing_data(train_data, subject_medcat_names)")
    train_df = preprocessing_data(train_data, subject_medcat_names)

    print("START preprocessing_data(train_data, subject_medcat_names)")
    test_df = preprocessing_data(test_data, subject_medcat_names)

    print("START processing_df(train_df)")
    df_train = processing_df(train_df, subject_id_to_name)

    print("START processing_df(test_df)")
    df_test = processing_df(test_df, subject_id_to_name)
    
    return df_train, df_test



def generate_prompt(df_train, df_test):
    ''' Generate Prompt'''
    prompt_format1_cate = """%s has %d conditions. Given the note, what is/are the conditions? Note: %s"""
    prompt_format2_cate = """Determine %s 's %d conditions. Note: %s"""
    prompt_format3_cate = """Based on the patient note for %s, identify the %d medical conditions present. Note: %s"""
    prompt_format4_cate = """Review the medical notes for %s indicating a total of %d health conditions. What health issues have been diagnosed? Note: %s"""
    prompt_format5_cate = """For patient %s, the medical record lists %d conditions. what is/are the conditions? Note: %s"""

    prompts_cate = [prompt_format1_cate, prompt_format2_cate, prompt_format3_cate, prompt_format4_cate, prompt_format5_cate]

    def gen_prompt_cate(element):
        prompt_format = prompts_cate[random.randint(0, len(prompts_cate)-1)]
        return DatasetDict({'prompt': prompt_format%(element['name'], element['num'],element['note']), "completion":element['condition']})

    print("START Dataset.from_dict(df_train)")
    datadict_train = Dataset.from_dict(df_train)

    print("START Dataset.from_dict(df_test)")
    datadict_test = Dataset.from_dict(df_test)

    print("START datadict_train.map(gen_prompt_cate")
    train_cate = datadict_train.map(gen_prompt_cate, remove_columns=datadict_train.column_names)
    train_dataset = train_cate

    print("START datadict_test.map(gen_prompt_cate,")
    test_cate = datadict_test.map(gen_prompt_cate, remove_columns=datadict_test.column_names)
    test_dataset = test_cate
    
    return train_dataset, test_dataset


def load_model(args):
    ''' Load Model'''
    if args.model == "opt":
        model_name = "facebook/opt-6.7b"
    elif args.model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    else:
        print("CHECK MODL NAME AGAIM")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=args.bit8,
        device_map="auto",
        torch_dtype="auto" #add
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    for param in model.parameters():    
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)#float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    

    ##################Apply LoRA
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
    #################Training
    response_template = " ### Condition:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    torch.cuda.empty_cache()
    
    for param in model.parameters(): # model.generate -> RuntimeError: expected scalar type Float but found Half
    # Check if parameter dtype is  Half (float16)
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)
            

    training_args=TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_step,
        report_to = "wandb",
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.lr_rate,
        lr_scheduler_type = args.lr_schedular, #add
        fp16 = args.fp16,
        do_eval = True,
        do_train = True,
    #    bf16=True,
        logging_steps = args.logging_steps,
        output_dir = args.output_path
    )

    if args.use_collator:
        trainer = SFTTrainer(
            model = model,
            args = training_args,
            max_seq_length = min(args.max_seq, tokenizer.model_max_length),
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            formatting_func = formatting_prompts_func,
            data_collator =collator,
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
        #    data_collator =collator,
            peft_config=peft_config,
        )

    model.config.use_cache = args.model_use_cache 
    
    trainer.train()

    trainer.save_model(args.output_path)
    
    return model, tokenizer


def generate_prompt_for_test(args):
    random.seed(42)
    path = f'data/{args.model}/'
    prompt_list = pd.read_csv(path + 'llama2_prompts.csv')
    sample = pd.read_csv(path + f'sample_{args.sample}_4000.csv')

    prompts = list(prompt_list['Prompt']) 
    
    filled_prompts = []
    for index, row in sample.iterrows():
        sentence = prompts[random.randint(0, len(prompts)-1)]
        filled_prompt = sentence.replace("{name}", row['name'])
        # Append a tuple of the filled sentence and the condition
        filled_prompts.append((filled_prompt, row['condition'],row['SUBJECT_ID']))
    
    random.shuffle(filled_prompts)

    return filled_prompts


def test(args):
    # "outputs_0424"
    set_seed(42)
    print("========Test==========")
    
    if args.model == "opt":
        model_name = "facebook/opt-6.7b"
    elif args.model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    #model = model.bfloat16() -> RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
    model=PeftModel.from_pretrained(model, args.output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.output_path)#.to('cuda')

    test_prompt = generate_prompt_for_test(args)
    model.eval()
    results = []
    for idx, i in tqdm(enumerate(test_prompt),desc="Generating: ", position=0, ncols=150 ):
        encodings = tokenizer(test_prompt[idx][0], return_tensors="pt", padding=True).to('cuda')

        generation_outputs = model.generate(
            **encodings,
            max_new_tokens=500,
            num_beams=3,
            no_repeat_ngram_size=5,
            temperature=1,
        )

        output = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)

        results.append((test_prompt[idx][2],output,test_prompt[idx][1]))
        print(f'{idx}: {output,test_prompt[idx][1]}')


    final_result = pd.DataFrame(results, columns=['SUBJECT_ID','output', 'condition'])

    final_result.to_csv(f'{args.model}_{args.output_path}_{args.sample}_{args.run_name}.csv', index=False)
    return final_result


