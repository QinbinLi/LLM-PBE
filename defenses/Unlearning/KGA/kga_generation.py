from lightning.fabric import Fabric
from accelerate import Accelerator
import copy
import os
import logging
import numpy as np
import math
import random
import torch
import time
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, BartModel, AutoConfig
from tqdm import tqdm
from itertools import chain
from safetensors.torch import load_file

from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

from .arg import parse_args
from .data_ import TRANS, get_dataLoader
from .run_generation import seed_everything, metric_evaluate, diff_evaluate, InverseSquareRootSchedule

from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
)

import sys
sys.path.insert(0, '/home/junyi/unlearn/LLM-PBE/')



from data.echr import EchrDataset
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("Model")

# fabric = Fabric(accelerator='cuda', precision='bf16-mixed')
# fabric.launch()

def dev_loop(args, dataloader, model):
    model.eval()
    total_loss = 0.0
    total_num = 0.0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Evaluating...'):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            outputs = model(**batch_data)
            loss = outputs.loss
            total_loss += loss.item()
            total_num += 1
    return total_loss/total_num


def more_args():
    _, parser = parse_args()

    parser.add_argument("--new_model_dir", default=None, type=str)
    parser.add_argument("--forget_model_dir", default=None, type=str)
    parser.add_argument("--train_model_dir", default=None, type=str)

    parser.add_argument("--num_train_updates", default=2000, type=int, help="Total number of updates to perform.")
    parser.add_argument("--do_unlearn", action="store_true", help="Whether to run unlearning.")

    parser.add_argument("--forget_file", default=None, type=str, help="The input forget set.")
    parser.add_argument("--new_file", default=None, type=str, help="The input new set.")

    parser.add_argument("--retain_loss_ratio", default=0.1, type=float, help="Ratio for remaining loss.")
    parser.add_argument("--stop_value", default=None, type=float, help="")
    parser.add_argument("--stop_portion", default=None, type=float, help="")

    return parser.parse_args()


# Preprocessing the ids
def load_ids(filename: str):
    with open(filename, 'r') as f:
        ids = [int(line.strip()) for line in f.readlines()]
    return ids

device_model = 'cuda:0'
device_train_model = 'cuda:1'
device_forget_model = 'cuda:2'
device_new_model = 'cuda:3'

def unlearn(
        args, train_dataset, dev_dataset, forget_dataset, new_dataset,
        train_model, forget_model, new_model, model, tokenizer):
    train_dataloader = get_dataLoader(args, train_dataset, model, tokenizer, shuffle=True)
    forget_dataloader = get_dataLoader(args, forget_dataset, model, tokenizer, shuffle=True)
    new_dataloader = get_dataLoader(args, new_dataset, model, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, model, tokenizer, shuffle=False)
    

    accelerator = Accelerator(gradient_accumulation_steps=1)
    logger.info(accelerator.state)
    accelerator.wait_for_everyone()

    

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = InverseSquareRootSchedule(
        warmup_init_lr=-1,
        warmup_updates=args.warmup_steps,
        lr=args.learning_rate,
        optimizer=optimizer
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
    )

    # Train!
    logger.info("***** Running unlearning *****")
    logger.info(f"Train: {len(train_dataset)}, Forget: {len(forget_dataset)}, New: {len(new_dataset)}, Dev: {len(dev_dataset)}")
    logger.info(f"Num Updates/Epochs - {args.num_train_updates}/{args.num_train_epochs}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))


    model.to(device_model)
    train_model.to(device_train_model)
    forget_model.to(device_forget_model)
    new_model.to(device_new_model)

    # model, optimizer = fabric.setup(model, optimizer)
    # train_dataloader = fabric.setup_dataloaders(train_dataloader)
    
    total_updates = 0
    epoch_num = 0
    stop_value = args.stop_value
    stop = False
    forget_iterator = iter(forget_dataloader)
    new_iterator = iter(new_dataloader)
    while epoch_num < args.num_train_epochs and not stop:
        model.train()
        for total_steps, batch_data in enumerate(tqdm(train_dataloader, desc=f'Training epoch {epoch_num}'), start=1):
            batch_data_model = {k: v.to(model.device) for k, v in batch_data.items()}
            batch_data_train_model = {k: v.to(train_model.device) for k, v in batch_data.items()}

            outputs = model(**batch_data_model) # A_*, the model that trained
            tgt_outputs = train_model(**batch_data_train_model) # A_D, the model that before unlearned

            pred_logits = F.log_softmax(outputs.logits, dim=-1).to(model.device)
            tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach().to(model.device)

            loss_remain = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean') # Computer Lr based on Eq. 4
            # (loss_remain * args.retain_loss_ratio / args.update_freq).backward()
            # fabric.backward(loss_remain * args.retain_loss_ratio / args.update_freq)
            accelerator.backward(loss_remain * args.retain_loss_ratio / args.update_freq)

            if total_steps % args.update_freq == 0:
                total_updates += 1

                try:
                    batch_forget_model = next(forget_iterator)
                except StopIteration:
                    forget_iterator = iter(forget_dataloader)
                    batch_forget_model = next(forget_iterator)

                try:
                    batch_new = next(new_iterator)
                except StopIteration:
                    new_iterator = iter(new_dataloader)
                    batch_new = next(new_iterator)

                batch_forget_model = {k: v.to(model.device) for k, v in batch_forget_model.items()}
                batch_forget_forget_model = {k: v.to(forget_model.device) for k, v in batch_forget_model.items()}
                batch_new_train_model = {k: v.to(train_model.device) for k, v in batch_new.items()}
                batch_new_new_model = {k: v.to(new_model.device) for k, v in batch_new.items()}

                outputs = model(**batch_forget_model)
                tgt_outputs = forget_model(**batch_forget_forget_model)

                pred_logits = F.log_softmax(outputs.logits, dim=-1).to(model.device)
                tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach().to(model.device)
                loss_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean')

                outputs = train_model(**batch_new_train_model)
                tgt_outputs = new_model(**batch_new_new_model)
                
                pred_logits = F.log_softmax(outputs.logits, dim=-1).detach().to(train_model.device)
                tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach().to(train_model.device)
                tgt_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean')
                loss_align = torch.abs(loss_align - tgt_align.item()) # Compute La based on Eq. 3

                # loss_align.backward()
                # fabric.backward(loss_align)
                accelerator.backward(loss_align)

                if stop_value is None and args.stop_portion is not None:
                    stop_value = loss_align.item() * args.stop_portion
                    logger.info(f'Set stop value as {stop_value} (Portion {args.stop_portion})')

                optimizer.step()
                lr_scheduler.step(total_updates)
                optimizer.zero_grad()

                total_loss = loss_align.item() + args.retain_loss_ratio * loss_remain.item() # La + a * Lr

                
                if total_updates % 20 == 0:
                    logger.info(f'Train Step {total_updates}, total loss: {total_loss:>7f}, align loss {loss_align.item():>7f}, retain loss {loss_remain.item():>7f}')
                    
                    # if stop_value is not None and loss_align.item() <= stop_value: # compute current GAP, if GAP <= stop_value, stop
                    #     stop = True
                    #     logger.info(f'current gap is smaller than {stop_value}, training stopped')
                    #     break
                    model.train()
        
        epoch_num += 1
        
        logger.info(f"Epoch {epoch_num} finished, evaluating ...")
        dev_loss = dev_loop(args, dev_dataloader, model) # Evaluate
        logger.info(f'Loss (evaluation) {dev_loss:0.4f}')
        forget_loss = dev_loop(args, get_dataLoader(args, forget_dataset, model, tokenizer, shuffle=False), model)
        logger.info(f'Loss (forget) {forget_loss:0.4f}')
        
        logger.info(f"Epoch {epoch_num} finished, saving to epoch_{epoch_num}_update_{total_updates}_dev_loss_{dev_loss:0.4f}_forget_loss_{forget_loss:0.4f}_weights.bin")
        save_weight = f'epoch_{epoch_num}_update_{total_updates}_dev_loss_{dev_loss:0.4f}_forget_loss_{forget_loss:0.4f}_weights.bin'
        torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))


if __name__ == '__main__':
    args = more_args()

    config = LoraConfig(r=8,
                             lora_alpha=16,
                             lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             bias='none',
                             task_type="CAUSAL_LM")


    if args.do_unlearn:
        assert args.new_model_dir is not None
        assert args.forget_model_dir is not None
        assert args.train_model_dir is not None
    assert args.output_dir is not None
    if os.path.exists(args.output_dir):
        files = os.listdir(args.output_dir)
        if any([('.bin' in f) for f in files]):
            raise ValueError(f'Output directory ({args.output_dir}) already exists saved models.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)

    # Load pretrained model and tokenizer
    logger.info(f'Loading model ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, trust_remote_code=True)

    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    
    new_model = copy.deepcopy(model)
    logger.info(f'loading new model from {args.new_model_dir} ...')
    new_model.load_state_dict(load_file(args.new_model_dir))
    new_model.eval()
    for p in new_model.model.parameters():
        p.requires_grad = False


    forget_model = copy.deepcopy(model)
    logger.info(f'loading forget model from {args.forget_model_dir} ...')
    forget_model.load_state_dict(load_file(args.forget_model_dir))
    forget_model.eval()
    for p in forget_model.model.parameters():
        p.requires_grad = False


    logger.info(f'loading trained model from {args.train_model_dir} ...')
    # model.load_state_dict(torch.load(args.train_model_dir))
    train_model = AutoModelForCausalLM.from_pretrained(args.train_model_dir, trust_remote_code=True)
    train_model = get_peft_model(train_model, config)
    print(train_model.print_trainable_parameters())
    
    train_model.eval()
    for p in train_model.model.parameters():
        p.requires_grad = False
    
    
    
    # Unlearning
    if args.do_unlearn:
        # train_dataset = TRANS(args.train_file, args.source, args.target)
        # dev_dataset = TRANS(args.dev_file, args.source, args.target)
        # forget_dataset = TRANS(args.forget_file, args.source, args.target)
        # new_dataset = TRANS(args.new_file, args.source, args.target)
        begin_time = time.time()

        raw_datasets = EchrDataset(data_path="/home/junyi/unlearn/LLM-PBE/data/echr", pseudonymize=True, mode='undefended').raw_datasets # 没有做任何防御，原始数据

        forget_ids = load_ids(args.forget_file)
        new_ids = load_ids(args.new_file)
        train_ids = list(set(range(len(raw_datasets['train']))) - set(forget_ids) - set(new_ids)) # exclude forget and new

        train_dataset = raw_datasets["train"].select(train_ids) # TODO: remove this sample
        new_dataset = raw_datasets['train'].select(new_ids)
        forget_dataset = raw_datasets['train'].select(forget_ids)
        dev_dataset = raw_datasets["validation"]

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]


        def preprocess(datasets, block_size = 512, num_proc=4, desc='?'):
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name])

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc=f"Running tokenizer on dataset {desc}",
            )

            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc=f"Grouping texts in chunks of {block_size} on {desc}",
            )
            return lm_datasets

        train_dataset = preprocess(train_dataset, desc='training set')
        new_dataset = preprocess(new_dataset, num_proc=1, desc='new set')
        forget_dataset = preprocess(forget_dataset, num_proc=1, desc='forget set')
        dev_dataset = preprocess(dev_dataset, num_proc=1, desc='dev set')

        unlearn(args, train_dataset, dev_dataset, forget_dataset, new_dataset, train_model, forget_model, new_model, model, tokenizer)
        logger.info(f'Total used time: {(time.time() - begin_time) / 60} minutes!')
    else:
        raise NotImplementedError

