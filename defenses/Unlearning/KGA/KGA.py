"""Integration of KGA Unlearn.

"""
import json
import logging
import math
import os
import random
from itertools import chain
from typing import Union
from .common import load_ids
import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import Repository, create_repo
from peft import (PeftType, PromptTuningConfig, PromptTuningInit, TaskType,
                  get_peft_model)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (MODEL_MAPPING, AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, PreTrainedModel, SchedulerType,
                          default_data_collator, get_scheduler)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from data.echr import EchrDataset
from defenses.Unlearning import UnlearningBase

logger = get_logger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt"
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class KGAUnlearn(UnlearningBase):

    def __init__(
        self,
        model: PreTrainedModel,
        model_f: PreTrainedModel,
        model_n: PreTrainedModel,
        file_forget_ids: str,
        file_assist_ids: str,
        checkpoint:str = "meta-llama/Llama-2-7b-chat-hf",
        output_dir:str = "kga_unlearned",
    ):
        """
        Args:
            model: the model DIRECTORY that you want to unlearn
            model_f: safetensors model path, the model trained on Df (the forget set)
            model_n: safetensors model path, the model trained on Dn (the new set, a small set of extra data Dn, where Dn ∩ D = ∅) to assist the unlearning process.
        """
        # super().__init__(model, data, prompt, params)

        self.model = model
        self.model_f = model_f
        self.model_n = model_n
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.file_forget_ids = file_forget_ids
        self.file_assist_ids = file_assist_ids

    def execute(self):
        os.system(f"""accelerate launch kga_generation.py \
    --output_dir {self.output_dir} \
    --new_model_dir {self.model_n} \
    --forget_model_dir {self.model_f} \
    --train_model_dir {self.model} \
    --model_checkpoint {self.checkpoint} \
    --forget_file {self.file_forget_ids} \
    --new_file {self.file_assist_ids} \
    --do_unlearn \
    --retain_loss_ratio 0.3 \
    --batch_size 2 \
    --update_freq 2 \
    --learning_rate 5e-5 \
    --num_train_updates 12000 \
    --num_train_epochs 50 \
    --stop_value 0.05 \
    --warmup_steps 1000 \
    --weight_decay 0.0001 \
    --lr_schedule inverse_sqrt \
    --beam 5 | tee -a unlearn_kga.log""")

"""
Only use this help for training A_f and A_n.

(as defined in paper, we need to train A_f and A_n on D_f and D_n respectively.)
"""
class KGAHelper:

    def __init__(self, config: AutoConfig, model: PreTrainedModel, tokenizer: AutoTokenizer, raw_datasets: Union[DatasetDict, Dataset], checkpoint: str):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.raw_datasets = raw_datasets
        self.checkpoint = checkpoint

    def train_model_n(self,
                      file_assist_ids: str,
                      num_train_epochs: int) -> PreTrainedModel:
        # Train model on new set. (The Dn dataset, a small set of extra data Dn, where Dn ∩ D = ∅) to assist the unlearning process.
        self.train(
            checkpoint=self.checkpoint,
            model=self.model,
            config=self.config,
            tokenizer=self.tokenizer,
            raw_datasets=self.raw_datasets,
            file_assist_ids=file_assist_ids,
            num_train_epochs=num_train_epochs,
            output_dir="kga_output_n",
        )

    def train_model_f(self,
                      file_forget_ids: str,
                      num_train_epochs: int) -> PreTrainedModel:
        # Train model on forget set
        self.train(
            checkpoint=self.checkpoint,
            model=self.model,
            config=self.config,
            tokenizer=self.tokenizer,
            raw_datasets=self.raw_datasets,
            file_assist_ids=file_forget_ids,
            num_train_epochs=num_train_epochs,
            output_dir="kga_output_f",
        )

    def train(self,
              model: AutoModelForCausalLM,
              config: AutoConfig,
              tokenizer: AutoTokenizer,
              raw_datasets: Union[DatasetDict, Dataset],
              file_assist_ids: str,
              output_dir: str = None,
              checkpoint: str = None,
              seed: int = None,
              num_train_epochs: int = 50,
              block_size: int = 512,
              gradient_accumulation_steps: int = 1,
              max_train_steps: int = None,
              num_warmup_steps: int = 0,
              per_device_eval_batch_size: int = 4,
              per_device_train_batch_size: int = 4,
              preprocessing_num_workers: int = 8,
              overwrite_cache: bool = False,
              checkpointing_steps: int = 500,
              resume_from_checkpoint: str = None,
              lr_scheduler_type: str = "linear",
              learning_rate: float = 5e-5,
              weight_decay: float = 0.0) -> PreTrainedModel:
        """

        Args:
          lr_scheduler_type (str): The learning rate scheduler type. Default is "linear". You can choose from "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"

        """
        assert lr_scheduler_type in [
            "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]

        # Training on the Dn dataset (a small set of extra data Dn, where Dn ∩ D = ∅) to assist the unlearning process.

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we"re using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {}

        accelerator = Accelerator(gradient_accumulation_steps=1,
                                  **accelerator_log_kwargs)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if seed is not None:
            set_seed(seed)

        # Handle the repository creation
        if accelerator.is_main_process and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        accelerator.wait_for_everyone()

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        # using peft

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
            tokenizer_name_or_path=checkpoint,
        )
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]



        train_ids = load_ids(file_assist_ids)
        raw_datasets["train"] = raw_datasets["train"].select(train_ids)

        len_examples = len(raw_datasets["train"])
        print(f"Selected {len_examples} examples for training.")

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k]))
                for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i:i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=per_device_train_batch_size)
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps *
            gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps /
                                     num_update_steps_per_epoch)

        # Train!
        total_batch_size = per_device_train_batch_size * \
            accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps),
                            disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if resume_from_checkpoint:
            if resume_from_checkpoint is not None or resume_from_checkpoint != "":
                checkpoint_path = resume_from_checkpoint
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                # Sorts folders by date modified, most recent checkpoint is the last
                path = dirs[-1]
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(
                    training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace(
                    "step_", "")) * gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, num_train_epochs):
            model.train()

            if resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(
                    train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(
                    tqdm(active_dataloader, desc=f"Training epoch {epoch}")):
                # print("Accumulate step:", step)
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if output_dir is not None:
                            output_dir = os.path.join(output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= max_train_steps:
                    break

            print("Evaluation ...")
            model.eval()
            losses = []
            for step, batch in enumerate(
                    tqdm(eval_dataloader, desc=f"Evaluating epoch {epoch}")):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(
                    accelerator.gather_for_metrics(
                        loss.repeat(per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(
                f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
            )

            # Save the model checkpoint each epoch

            accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))

        # Save the final model checkpoint
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
                json.dump({"perplexity": perplexity}, f)
