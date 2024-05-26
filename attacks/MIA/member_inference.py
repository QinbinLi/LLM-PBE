import torch, os
import zlib
from attacks.AttackBase import AttackBase
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from enum import Enum
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from models.ft_clm import FinetunedCasualLM


class MIAMetric(Enum):
    """Metrics are from https://www.usenix.org/system/files/sec21-carlini-extracting.pdf."""
    LOSS = "loss"
    PPL = "perplexity"  # the perplexity of the largest GPT-2 model.
    REFER = "refer"  # the ratio of log-perplexities of the largest GPT-2 model and the reference model.
    # SMALL = "small"  # the ratio of log-perplexities of the largest GPT-2 model and the Small GPT-2 model.
    # Medium = "medium"  # the ratio as above, but for the Medium GPT-2
    ZLIB = "zlib"  # the ratio of the (log) of the GPT-2 perplexity and the zlib entropy (as computed by compressing the text).
    LOWER_CASE = "lowercase"  # the ratio of perplexities of the GPT-2 model on the original sample and on the lowercased sample
    WINDOW = "window"  #  the minimum perplexity of the largest GPT-2 model across any sliding window of 50 tokens.

    LIRA = "lira"  # https://arxiv.org/pdf/2203.03929.pdf
    NEIGHBOR = "neighbor"   # https://aclanthology.org/2023.findings-acl.719.pdf
    MIN_K_PROB = "min_k_prob"   # https://arxiv.org/pdf/2310.16789.pdf

class MemberInferenceAttack(AttackBase):
    """Membership Inference Attack (MIA).

    Note MIA is often used with data extraction to find the training samples in 
    generated samples. For this purpose, top-score samples will be selected.

    Reference implementation: https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    """
    def __init__(self, metric: MIAMetric, ref_model=None, n_neighbor=50):
        # self.extraction_prompt = ["Tell me about..."]  # TODO this is just an example to extract data.
        self.metric = metric
        self.ref_model = ref_model
        self.n_neighbor = n_neighbor
    
    @torch.no_grad()
    def _get_score(self, model: FinetunedCasualLM, text: str):
        """Return score. Smaller value means membership."""
        if self.metric == MIAMetric.PPL:
            ppl = model.evaluate_ppl(text)
            score = ppl
        elif self.metric == MIAMetric.LOSS:
            loss = model.evaluate(text)
            score = loss
        elif self.metric == MIAMetric.LOWER_CASE:
            ppl = model.evaluate_ppl(text)
            ref_ppl = model.evaluate_ppl(text.lower())
            score = ppl / ref_ppl
        elif self.metric == MIAMetric.WINDOW:
            # ppl = model.evaluate_ppl(text)
            assert model.tokenizer is not None
            input_ids = model.tokenizer(text, return_tensors='pt', 
                                        truncation=True, 
                                        max_length=model.max_seq_len).input_ids
            win_size = 50
            if len(input_ids) > win_size:
                ppls = []
                for idx in range(len(input_ids)- win_size):
                    _ppl = model.evaluate_ppl(input_ids[idx, idx+win_size], tokenized=True)
                    ppls.append(_ppl.item())
                score = np.min(ppls)
            else:
                score = model.evaluate_ppl(input_ids, tokenized=True)
        elif self.metric == MIAMetric.REFER:
            ppl = model.evaluate_ppl(text)
            ref_ppl = self.ref_model.evaluate_ppl(text)
            score = np.log(ppl) / np.log(ref_ppl)
        elif self.metric == MIAMetric.LIRA:
            # https://arxiv.org/pdf/2203.03929.pdf
            ppl = model.evaluate_ppl(text)
            ref_ppl = self.ref_model.evaluate_ppl(text)
            # score = np.log(ref_ppl) - np.log(ppl)
            score = np.log(ppl) - np.log(ref_ppl)
        elif self.metric == MIAMetric.NEIGHBOR:
            assert self.ref_model is not None, 'Neighborhood MIA requires a reference model'
            neighbor_avg = 0
            neighbors = self.ref_model.generate_neighbors(text, n=self.n_neighbor)
            for neighbor in neighbors:
                neighbor_avg += model.evaluate(neighbor)
            neighbor_avg /= len(neighbors)
            score = model.evaluate(text) - neighbor_avg
        elif self.metric == MIAMetric.ZLIB:
            ppl = model.evaluate_ppl(text)
            num_bits = len(zlib.compress(text.encode())) * 8
            score = ppl / num_bits
        elif self.metric == MIAMetric.MIN_K_PROB:
            # Get logits from model
            input_ids = model.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=model.max_seq_len).cuda()
            with torch.no_grad():
                outputs = model._lm(input_ids, labels=input_ids)
            logits = outputs[1]

            # Apply softmax to the logits to get probabilities
            probabilities = torch.nn.functional.log_softmax(logits, dim=-1).cpu().data
            all_prob = []
            input_ids_processed = input_ids[0][1:]
            for i, token_id in enumerate(input_ids_processed):
                probability = probabilities[0, i, token_id].item()
                all_prob.append(probability)

            # Calculate Min-K% Probability
            k_length = int(len(all_prob) * 0.10)  # TODO: For now, K is hard-coded as 10%
            topk_prob = np.sort(all_prob)[:k_length]
            score = -np.mean(topk_prob).item()
        else:
            raise NotImplementedError(f"{self.metric}")
        return score
    
    def execute(self, model, train_set, test_set, cache_file=None, resume=False):
        """Compute scores for texts.

        Parameters:
        - memberships: A list of 0,1. 0 means non-member and 1 means member.

        Returns:
        - scores: the scores of membership.
        """
        model._lm.eval()
        if resume:
            if os.path.exists(cache_file):
                print(f"resume from {cache_file}")
                loaded = torch.load(cache_file)
                results = loaded['results']
                print(f"resume: i={loaded['i']}, member={loaded['member']}")
            else:
                print(f"WARN: Cann't resume. Not found {cache_file}.")
                resume = False
                results = defaultdict(list)
        else:
            results = defaultdict(list)
        if resume:
            if loaded['member'] != 1:
                print(f"Train set has been evaluated.")
                resume_i = len(train_set)
            else:
                resume_i = loaded['i']
                print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
        member = 1
        for i, sample in enumerate(tqdm(train_set)):
            if i <= resume_i:
                continue
            score = self._get_score(model, sample['text'])
            results['score'].append(score)
            results['membership'].append(member)
            if (i+1) % 100 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Train avg score: {np.mean(np.array(results['score']))}")
        
        test_scores = []
        member = 0
        
        if resume and loaded['member'] == 0:
            resume_i = loaded['i']
            print(f"Resume from {resume_i+1}/{len(test_set)}")
        else:
            resume_i = -1
        for i, sample in enumerate(tqdm(test_set)):
            if i <= resume_i:
                continue
            score = self._get_score(model, sample['text'])
            results['score'].append(score)
            test_scores.append(score)
            results['membership'].append(0)
            if (i+1) % 30 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Test avg score: {np.mean(np.array(test_scores))}")
        torch.save({'results': results, 'i': -1, 'member': -1}, cache_file)
        return results

    def evaluate(self, results):
        # results['score']
        score_dict = {}
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        # # follow https://arxiv.org/pdf/2203.03929.pdf
        # threshold = np.quantile(results['score'][results['membership']==0], 0.9)
        threshold = np.mean(results['score'][results['membership']==0])
        score_dict['nonmember_score'] = np.mean(results['score'][results['membership']==0])
        score_dict['member_score'] = np.mean(results['score'][results['membership']==1])
        # for computing AUC, you can use any threshold.
        # threshold = np.quantile(results['score'], 0.5)
        results['score'] -= threshold
        # this is for the ease of using roc_auc_score, which is equivalent to varying threshold.
        # results['score'] = 1. - 1 / (1 + np.exp(- results['score']))
        # NOTE score has to be reversed such that lower score implies membership.
        score_dict['acc'] = accuracy_score(results['membership'], results['score'] < 0)
        score_dict['auc'] = roc_auc_score(results['membership'], - results['score'])
        fpr, tpr, thresholds = roc_curve(results['membership'], - results['score'])
        score_dict[r'TPR@0.1%FPR'] = None
        for fpr_, tpr_, thr_ in zip(fpr, tpr, thresholds):
            if fpr_ > 0.001:
                score_dict[r'TPR@0.1%FPR'] = tpr_
                break
        return score_dict
