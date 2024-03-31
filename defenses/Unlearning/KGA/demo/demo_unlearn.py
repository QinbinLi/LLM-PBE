import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data.echr import EchrDataset
from defenses.Unlearning.KGA import KGAHelper, KGAUnlearn

curr_path = os.path.abspath(__file__)

raw_datasets = EchrDataset(
    data_path="data/echr", pseudonymize=True,
    mode="undefended").raw_datasets  # undefended dataset.

# raw_datasets['validation'] = raw_datasets['validation'].shuffle().select([i for i in range(1000)])

checkpoint = "meta-llama/Llama-2-7b-chat-hf"
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, config=config, trust_remote_code=True)


helper = KGAHelper(config=config,
                   model=model,
                   tokenizer=tokenizer,
                   raw_datasets=raw_datasets,
                   checkpoint=checkpoint,)

file_forget_ids = os.path.join(curr_path, "ids_forget_warsaws.txt")
file_assist_ids = os.path.join(curr_path, "ids_assist_hungary.txt")

# file_assist_ids: a file containing the ids of the training set, one id per line
helper.train_model_n(
    file_assist_ids=file_assist_ids,
    num_train_epochs=3,
)

# file_forget_ids: a file containing the ids of the forget set, one id per line
helper.train_model_f(
    file_forget_ids=file_forget_ids,
    num_train_epochs=3,
)


kga = KGAUnlearn(
    model   = '/data/junyi/unlearn_original',
    model_n = '/home/junyi/unlearn/LLM-PBE/unlearn_assist_hungary/epoch_49/model.safetensors',
    model_f = '/home/junyi/unlearn/LLM-PBE/unlearn_forget_warsaws/epoch_49/model.safetensors',
    file_assist_ids=file_assist_ids,
    file_forget_ids=file_forget_ids,
)
