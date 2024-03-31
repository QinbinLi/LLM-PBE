# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy as np
import transformers
from tqdm import tqdm
import wandb

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.privacy_attack import PrivacyAttack, ExtractionAttack, ReconstructionAttack
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_dict_highlighted
from pii_leakage.utils.set_ops import intersection


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            DatasetArgs,
                                            AttackArgs,
                                            EvaluationArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evaluate(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()
    
    wandb.init(project='pii-leakage', name='evaluate')

    print_dict_highlighted(vars(attack_args))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Load the baseline model (publicly pre-trained).
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_args.peft = 'none'
    baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args, env_args=env_args)
    eval_dataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"), ner_args=ner_args, env_args=env_args)


    if not env_args.skip_ppl_eval:
        lm_ppl = lm.perplexity(eval_dataset["text"])
        print(f"loaded lm ppl = {lm_ppl}")
        baseline_lm_ppl = baseline_lm.perplexity(eval_dataset["text"])
        print(f"baseline lm ppl = {baseline_lm_ppl}")

        wandb.summary['lm_ppl'] = lm_ppl
        wandb.summary['baseline_lm_ppl'] = baseline_lm_ppl
    else:
        print(f"Skipped ppl eval. To eval ppl, use `--skip_ppl_eval=False`.")

    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"(Example) 20 Real PII: {real_pii.unique().mentions()[:20]}")

    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):
        # Compute Precision/Recall for the extraction attack.
        generated_pii: dict = attack.attack(lm)  # PII -> count
        print(f"LM Generated PII: {len(generated_pii)}")
        generated_baseline_pii: dict = attack.attack(baseline_lm)
        print(f"Baseline LM Generated PII: {len(generated_baseline_pii)}")

        # Remove baseline leakage
        leaked_piis = {}
        for pii in generated_pii:
            if pii not in generated_baseline_pii:
                leaked_piis[pii] = generated_pii[pii]
        
        if len(leaked_piis) > 0:
            real_pii_set = {v.text: -1 for v in real_pii.unique()}
            n_leaked = len(intersection(leaked_piis, real_pii_set))
            prec = n_leaked / len(leaked_piis)
            recall = n_leaked / len(real_pii_set)
            print(f"Precision: {100 * prec:.2f}%")
            print(f"Recall: {100 * recall:.2f}%")

            wandb.summary['extract prec'] = prec
            wandb.summary['extract recall'] = recall
        else:
            print(f"Not found PII")
            wandb.summary['extract prec'] = None
            wandb.summary['extract recall'] = None
    elif isinstance(attack, ReconstructionAttack):
        # Compute accuracy for the reconstruction/inference attack.
        idx = random.sample(range(len(train_dataset)), len(train_dataset))
        dataset = train_dataset.select(idx)  # dict with 'text': seq and 'entity_class': 'ListPII (as a str)'

        tagger = TaggerFactory.from_ner_args(ner_args, env_args=env_args)
        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            y_preds, y_trues = [], []
            all_accs = []
            for seq in dataset:
                if pbar.n > eval_args.num_sequences:
                    break

                # 1. Assert that the sequence has at least one PII
                pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
                pii = ListPII(data=[p for p in pii if len(p.text) > 3])  # min chars for PII
                if len(pii) == 0:
                    continue

                # 2. Randomly sample one target PII
                target_pii = random.sample(pii.mentions(), 1)[0]

                # 3. Replace the target PII with <T-MASK> and other PII with <MASK>
                target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
                for pii_mention in pii.mentions():
                    target_sequence = target_sequence.replace(pii_mention, '<MASK>')

                # 4. Randomly sample candidates
                assert eval_args.num_candidates <= len(real_pii.unique().mentions()), f"Not enough real candidates " \
                                                                                        f"({len(real_pii.unique().mentions())}) to accomodate candidate size ({eval_args.num_candidates})."
                candidate_pii = random.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [
                    target_pii]
                random.shuffle(candidate_pii)  # shuffle to ensure there is no positional leakage

                # 5. Run the reconstruction attack
                result = attack.attack(lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                predicted_target_pii = result[min(result.keys())]

                # 6. Evaluate baseline leakage
                baseline_result = attack.attack(baseline_lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                baseline_target_pii = baseline_result[min(baseline_result.keys())]

                if baseline_target_pii == predicted_target_pii:
                    # Baseline leakage because public model has the same prediction. Skip
                    continue

                y_preds += [predicted_target_pii]
                y_trues += [target_pii]

                acc = np.mean([1 if y_preds[i] == y_trues[i] else 0 for i in range(len(y_preds))])
                pbar.set_description(f"Evaluate Reconstruction: Accuracy: {100 * acc:.2f}%")
                pbar.update(1)
                all_accs.append(acc)
                wandb.log({'reconstruct acc': acc})
            wandb.summary['avg reconstruct acc'] = np.mean(all_accs)
    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
