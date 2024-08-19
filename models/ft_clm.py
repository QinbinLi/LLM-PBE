import transformers
import torch
import numpy as np
from heapq import nlargest
from transformers import AutoModelForCausalLM, AutoTokenizer
# from pii_leakage.utils.web import is_valid_url, download_and_unzip

from .LLMBase import LLMBase

class SamplingArgs:
    def __init__(self, prefix_length=50, suffix_length=50, do_sample=True, top_k=24, top_p=0.8, typical_p=0.9, temperature=0.58, repetition_penalty=1.04, zlib=False, context_window=4, high_conf=True):
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.zlib = zlib
        self.context_window = context_window
        self.high_conf = high_conf

class FinetunedCasualLM(LLMBase):
    """Huggingface Casual Language Models.

    Parameters:
    - model_path (str): The path/name for the desired langauge model.
        Supported models:
        1. llama2-7b, llama2-7b-chat: Find the model path at https://huggingface.co/LLM-PBE.
        2. gpt2, gpt2-large, gpt2-xl: The names for models on huggingface. Should manually download.
        3. Local path pointed to GPT2 model finetuned based on https://github.com/microsoft/analysing_pii_leakage.
        Analyzing Leakage of Personally Identifiable Information in Language Models. Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz and Santiago Zanella-Béguelin. Symposium on Security and Privacy (S&P '23). San Francisco, CA, USA.
    """
    def __init__(self, model_path=None, arch=None, max_seq_len=1024):
        if ':' in model_path:
            model_path, self.model_revision = model_path.split(':')
        else:
            self.model_revision = 'main'
        if arch is None:
            self.arch = model_path
            # if 'gpt2' in model_path:
            #     self.arch = 'gpt2'
            # elif 'llama-2-7b' in model_path.lower():
            #     self.arch = 'meta-llama/Llama-2-7b-hf'
            # else:
            #     raise NotImplementedError(f"model_path: {model_path}")
        else:
            self.arch = arch
        # default
        self.tokenizer_use_fast = True
        self.max_seq_len = max_seq_len
        self.verbose = True
        
        super().__init__(model_path=model_path)
    
    @property
    def tokenizer(self):
        return self._tokenizer

    def load_local_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = tokenizer.from_pretrained(self.arch,
                                                    use_fast=self.tokenizer_use_fast)
        if self.verbose:
            print(
                f"> Loading the provided {self.arch} checkpoint from '{model_path}'.")

        # if is_valid_url(model_path):
        #     model_path = download_and_unzip(model_path)
        try: 
            self._lm = model_cls.from_pretrained(model_path, return_dict=True, device_map='auto', revision=self.model_revision).eval()
        except:
            self._lm = model_cls.from_pretrained(model_path, return_dict=True, device_map='auto', revision=self.model_revision, offload_folder='./offload', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._lm.config.pad_token_id = self._lm.config.eos_token_id
        
    def query(self, text, new_str_only=False):
        """
        Query an open-source model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        # TODO pass the args into here. The params should be set according to PII-leakage.

        # Encode the text prompt and generate a response
        input_ids = self._tokenizer.encode(text, return_tensors='pt')
        # output = self.model.generate(input_ids)
        
        # Implement the code to query the open-source model
        output = self._lm.generate(
            input_ids=input_ids.to('cuda'),
            # attention_mask=attention_mask.to(self.env_args.device),
            max_new_tokens=self.max_seq_len,
            # max_length=min(self.n_positions, input_len + sampling_args.seq_len),
            # do_sample=sampling_args.do_sample,
            # top_k=sampling_args.top_k,
            # top_p=sampling_args.top_p,
            # output_scores=True,
            return_dict_in_generate=True,
            
        )

        # Decode the generated text back to a readable string
        if new_str_only:
            generated_text = self._tokenizer.decode(output.sequences[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            generated_text = self._tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return generated_text
        
    def evaluate(self, text, tokenized=False):
        """
        Evaluate an open-source model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - loss: The model's loss.
        """
        # TODO pass the args into here. The params should be set according to PII-leakage.
        if tokenized:
            input_ids = text
        else:
            # Encode the text prompt and generate a response
            input_ids = self._tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=self.max_seq_len)
            # output = self.model.generate(input_ids)
        
        # Implement the code to query the open-source model
        input_ids = input_ids.to('cuda')
        output = self._lm(
            input_ids=input_ids,
            labels=input_ids.clone(),
        )
        return output.loss.item()
        
    def evaluate_ppl(self, text, tokenized=False):
        """
        Evaluate an open-source model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - PPL: The model's perpelexity.
        """
        loss = self.evaluate(text, tokenized=tokenized)
        return np.exp(loss)

    def generate_neighbors(self, text, p=0.7, k=5, n=50):
        """
        For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements 
        at each position in the sequence and returning the top N neighboring sequences.

        https://aclanthology.org/2023.findings-acl.719.pdf
        """

        tokenized = self._tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_seq_len).input_ids.to('cuda')
        dropout = torch.nn.Dropout(p)

        seq_len = tokenized.shape[1]
        cand_scores = {}
        for target_index in range(1, seq_len):
            target_token = tokenized[0, target_index]
            
            # Embed the sequence
            if isinstance(self._lm, transformers.LlamaForCausalLM):
                embedding = self._lm.get_input_embeddings()(tokenized)
            elif isinstance(self._lm, transformers.GPT2LMHeadModel):
                embedding = self._lm.transformer.wte.weight[tokenized]
            else:
                raise RuntimeError(f'Unsupported model type for neighborhood generation: {type(self._lm)}')
            
            # Apply dropout only to the target token embedding in the sequence
            embedding = torch.cat([
                embedding[:, :target_index, :], 
                dropout(embedding[:, target_index:target_index+1, :]), 
                embedding[:, target_index+1:, :]
            ], dim=1)

            # Get model's predicted posterior distributions over all positions in the sequence
            probs = torch.softmax(self._lm(inputs_embeds=embedding).logits, dim=2)
            original_prob = probs[0, target_index, target_token].item()

            # Find the K most probable token replacements, not including the target token
            # Find top K+1 first because target could still appear as a candidate
            cand_probs, cands = torch.topk(probs[0, target_index, :], k + 1)
            
            # Score each candidate
            for prob, cand in zip(cand_probs, cands):
                if cand == target_token:
                    continue
                denominator = (1 - original_prob) if original_prob < 1 else 1E-6
                score = prob.item() / denominator
                cand_scores[(cand, target_index)] = score
        
        # Generate and return the neighborhood of sequences
        neighborhood = []
        top_keys = nlargest(n, cand_scores, key=cand_scores.get)
        for cand, index in top_keys:
            neighbor = torch.clone(tokenized)
            neighbor[0, index] = cand
            neighborhood.append(self._tokenizer.batch_decode(neighbor)[0])
        
        return neighborhood


class PeftCasualLM(FinetunedCasualLM):
    def load_local_model(self):
        super().load_local_model(self.arch)
        from peft.peft_model import PeftModel
        print(f"load peft module from {self.model_path}")
        try: 
            self._lm = PeftModel.from_pretrained(self._lm, self.model_path, device_map='auto')
        except:
            self._lm = PeftModel.from_pretrained(self._lm, self.model_path, device_map='auto', offload_folder='./offload')
        # self._lm = self._lm.merge_and_unload()


if __name__ == '__main__':
    # Testing purposes
    model = FinetunedCasualLM('gpt2')
    print(model.query(['hello. how are you?', 'what is your name?']))
    print("DONE")
