from tqdm import tqdm
import torch
import os
import numpy as np

class PromptLeakageSysPrompts:
    def __init__(self, category: str):
        
        self.system_prompts = []
        print(f"Select sys-prompt category: {category}")
        if category == 'GPTs':
            self.prompt_files = ['merged_GPTs.pth']
        elif category == 'blackfriday':
            self.prompt_files = ['dedup_BlackFriday-GPTs-Prompts.pth']
        elif category.startswith('blackfriday/'):
            subcat = category.split('/')[1]
            self.prompt_files = [f'blackfriday/dedup_{subcat}.pth']
        self.prompt_files = [os.path.join("data/prompt_leakage", f) for f in self.prompt_files]
        for f in self.prompt_files:
            self.system_prompts.extend(torch.load(f))
        print(f"Loaded {len(self.system_prompts)} prompts.")
    
    def random_select(self, n, seed=42):
        # self.random_subset_sys_prompts(n_sys_prompts)
        sys_prompt_rng = np.random.RandomState(seed)
        system_prompts = sys_prompt_rng.choice(self.system_prompts, n)
        print(f"Randomly select {len(system_prompts)} prompts by seed {seed}.")
        return system_prompts
    