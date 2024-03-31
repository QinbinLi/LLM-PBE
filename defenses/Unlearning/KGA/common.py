import os
import torch
import random
import numpy as np

# Preprocessing the ids
def load_ids(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        ids = [int(line.strip()) for line in f.readlines()]
    return ids

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# def metric_evaluate(args, dataset, model, tokenizer):
#     dataloader = get_dataLoader(args, dataset, model, tokenizer, shuffle=False)
#     logger.info('***** Running evaluation *****')

#     ppl = []
#     preds, labels = [], []
#     model.eval()
#     with torch.no_grad():
#         for batch_data in dataloader:  
#             batch_data = batch_data.to(args.device)
#             output = model(**batch_data)
#             ppl.append(math.exp(output.loss.item()))

#             generated_tokens = model.generate(
#                 batch_data["input_ids"],
#                 attention_mask=batch_data["attention_mask"],
#                 max_length=args.max_target_length,
#                 num_beams=args.beam,
#             ).cpu().numpy()
#             label_tokens = batch_data["labels"].cpu().numpy()

#             decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#             label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
#             decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

#             preds += [word_tokenize(pred.strip().lower()) for pred in decoded_preds]
#             labels += [[word_tokenize(label.strip().lower())] for label in decoded_labels]

#     logger.info(f'PPL: {(sum(ppl)/len(ppl)):0.4f}')
#     B1 = corpus_bleu(labels, preds, weights=(1.0, 0, 0, 0))
#     B2 = corpus_bleu(labels, preds, weights=(0.5, 0.5, 0, 0))
#     B3 = corpus_bleu(labels, preds, weights=(0.33, 0.33, 0.34, 0))
#     B4 = corpus_bleu(labels, preds, weights=(0.25, 0.25, 0.25, 0.25))
#     logger.info(f'BLEU1: {B1:0.4f}')
#     logger.info(f'BLEU2: {B2:0.4f}')
#     logger.info(f'BLEU3: {B3:0.4f}')
#     logger.info(f'BLEU4: {B4:0.4f}')

# def diff_evaluate(args, dataset, base_model, unlearn_model, tokenizer):
#     dataloader = get_dataLoader(args, dataset, model, tokenizer, batch_size=1, shuffle=False)
#     logger.info('***** Running evaluation *****')

#     ppl_dif1, ppl_dif2, ppl_dif3 = [], [], []
#     base_model.eval()
#     unlearn_model.eval()
#     with torch.no_grad():
#         for batch_data in dataloader:  
#             batch_data = batch_data.to(args.device)
#             output1 = base_model(**batch_data)
#             ppl1 = math.exp(output1.loss.item())
#             output2 = unlearn_model(**batch_data)
#             ppl2 = math.exp(output2.loss.item())

#             ppl_dif1.append(abs(ppl1 - ppl2))
#             ppl_dif2.append(abs(ppl1 - ppl2)/ppl1)
#             ppl_dif3.append(1.0 if ppl2 - ppl1 > 0 else 0.0)
#     ppl_dif1 = sum(ppl_dif1) / len(ppl_dif1)
#     ppl_dif2 = sum(ppl_dif2) / len(ppl_dif2)
#     ppl_dif3 = sum(ppl_dif3) / len(ppl_dif3)

#     logger.info(f'PPL Diff1: {ppl_dif1:0.4f}')
#     logger.info(f'PPL Diff2: {ppl_dif2:0.4f}')
#     logger.info(f'PPL Diff3: {ppl_dif3:0.4f}')




class InverseSquareRootSchedule(object):
    """From Fairseq
    """
    def __init__(self, warmup_init_lr, warmup_updates, lr, optimizer):
        super().__init__()

        self.optimizer = optimizer
        self.best = None

        warmup_end_lr = lr
        if warmup_init_lr < 0:
            warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first warmup_updates
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5

        # initial learning rate
        self.lr = warmup_init_lr
        self.set_lr(self.lr)

        self.warmup_init_lr = warmup_init_lr
        self.warmup_updates = warmup_updates

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.set_lr(self.lr)
        return self.lr

