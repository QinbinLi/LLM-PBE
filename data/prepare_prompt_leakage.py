"""Extract prompts from ./data"""
import os
import re
import torch
from tqdm import tqdm
import rapidfuzz

data_dir="./prompt_leakage"

def extract_leaked_GPTs(english_only=True):
    # Repo: Leaked-GPTs
    fld = "Leaked-GPTs/gpts"
    print(f"\n==== Extracting from {fld} ====")
    root_path = os.path.join(data_dir, f"{fld}")
    
    def extract(content: str):
        # Using regular expression to extract content between specific markers
        matches = re.findall(r'# System Prompt\s+```(.*?)```', content, re.DOTALL)
        return matches[0]
    
    file2prompts = read_md_files(root_path, extract)
    prompts = [v for k, v in file2prompts.items()]
    no_en_prompts = [p for p in prompts if not is_mostly_english(p)]
    prompts = [p for p in prompts if is_mostly_english(p)]
    print(prompts[0])
    
    fpath = os.path.join(data_dir, "Leaked-GPTs.pth")
    torch.save(prompts, fpath)
    print(f"\n>> Output {len(prompts)} prompts to: {fpath}")
    
    fpath = os.path.join(data_dir, "NE_Leaked-GPTs.pth")
    torch.save(no_en_prompts, fpath)
    print(f"\n>> Output {len(no_en_prompts)} non-english prompts to: {fpath}")


def extract_leaked_GPTs_no_pattern(english_only=True):
    # Repo: https://github.com/linexjlin/GPTs?tab=readme-ov-file
    fld = "GPTs/prompts"
    print(f"\n==== Extracting from {fld} ====")
    root_path = os.path.join(data_dir, f"{fld}")
    
    def extract(content: str):
        # Using regular expression to extract content between specific markers
        content = content.rstrip()
        if '```' not in content:
            print(f"There is no code block. Skip...")
            return None
        if content.endswith('```'):
            matches = re.findall(r'```markdown(.*?)```$', content.rstrip(), re.DOTALL)
        else:
            matches = re.findall(r'```markdown(.*?)$', content.rstrip(), re.DOTALL)
        if len(matches) < 1:
            # print(content)
            return None
        return matches[0]
    
    file2prompts = read_md_files(root_path, extract)
    prompts = [v for k, v in file2prompts.items()]
    no_en_prompts = [p for p in prompts if not is_mostly_english(p)]
    prompts = [p for p in prompts if is_mostly_english(p)]
    print(prompts[0])
    
    fpath = os.path.join(data_dir, "GPTs.pth")
    torch.save(prompts, fpath)
    print(f"\n>> Output {len(prompts)} prompts to: {fpath}")
    
    fpath = os.path.join(data_dir, "NE_GPTs.pth")
    torch.save(no_en_prompts, fpath)
    print(f"\n>> Output {len(no_en_prompts)} prompts to: {fpath}")

def extract_general_prompts(english_only=True):
    # Repo: BlackFriday-GPTs-Prompts
    fld = "BlackFriday-GPTs-Prompts"
    print(f"\n==== Extracting from {fld} ====")
    root_path = os.path.join(data_dir, f"{fld}")
    
    def extract(content: str):
        # Using regular expression to extract content between specific markers
        matches = re.findall(r'# Prompt\s+```(.*?)```\s+## Conversation', content, re.DOTALL)
        return matches[0]
    
    prompts = []
    NE_prompts = []
    categories = ['Academic', 'Business', 'Creative', 'Game', 'Job-Hunting', 'Marketing', 'Productivity-&-life-style', 'Programming']
    for cat in categories:
        print(f"--- {cat} ---")
        file_path = os.path.join(root_path, f"{cat}.md")
        md_filepaths = []
        with open(file_path, 'r') as file:
            pattern = r'\(\./gpts/(.*?)\.md\)'
            matches = re.findall(pattern, file.read())
            md_filepaths.extend([os.path.join("gpts", f"{f}.md") for f in matches])
        print(f"md_filepaths ({len(md_filepaths)}): {md_filepaths[0]}")
        
        file2prompts = read_md_files(root_path, extract, files=[f for f in md_filepaths if os.path.exists(os.path.join(root_path, f))])
        # for k, v in file2prompts.items():
        #     print(k, '\n', v)
        #     break
        _prompts = [v for k, v in file2prompts.items()]
        # if english_only:
        _no_en_prompts = [p for p in prompts if not is_mostly_english(p)]
        _prompts = [p for p in _prompts if is_mostly_english(p)]
        print(f"{cat}: {len(_prompts)} prompts")
        
        prompts.extend(_prompts)
        NE_prompts.append(_no_en_prompts)
        
        cat_fld = os.path.join(data_dir, "blackfriday")
        if not os.path.exists(cat_fld):
            os.makedirs(cat_fld)
        fpath = os.path.join(cat_fld, f"{cat}.pth")
        torch.save(_prompts, fpath)
        print(f"\n>> Output {len(_prompts)} prompts to: {fpath}")
        
        cat_fld = os.path.join(data_dir, "NE_blackfriday")
        if not os.path.exists(cat_fld):
            os.makedirs(cat_fld)
        fpath = os.path.join(cat_fld, f"{cat}.pth")
        torch.save(_no_en_prompts, fpath)
        print(f"\n>> Output NE {len(_no_en_prompts)} prompts to: {fpath}")
    
    # file2prompts = read_md_files(root_path, extract, files=[os.path.join("gpts", f"{f}.md") for f in md_filepaths])
    # for k, v in file2prompts.items():
    #     print(k, '\n', v)
    #     break
    # prompts = [v for k, v in file2prompts.items()]
    
    print()
    fpath = os.path.join(data_dir, "BlackFriday-GPTs-Prompts.pth")
    torch.save(prompts, fpath)
    print(f"\n>> Output {len(prompts)} prompts to: {fpath}")
    
    fpath = os.path.join(data_dir, "NE_BlackFriday-GPTs-Prompts.pth")
    torch.save(NE_prompts, fpath)
    print(f"\n>> Output {len(NE_prompts)} non-english prompts to: {fpath}")

# functions
def read_md_files(directory, extract_prompt, files=None) -> dict:
    md_files_content = {}
    if files is None:
        files = os.listdir(directory)
    # for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r') as md_file:
                prompt = extract_prompt(md_file.read())
                if prompt is not None and len(prompt) > 0:
                    md_files_content[file] = prompt
    return md_files_content

def is_mostly_english(input_string):
    # Using regular expressions to count the number of English and non-English characters
    english_chars = re.findall(r'[a-zA-Z]', input_string)
    non_english_chars = re.findall(r'[^a-zA-Z\s]', input_string)  # Exclude whitespace characters
    # print(len(english_chars), len(input_string))
    # if len(english_chars) < len(input_string) * 0.9:
    #     print(input_string, len(non_english_chars), len(english_chars), len(input_string))
    #     raise RuntimeError()

    # Comparing the counts to determine if the string is mostly English
    return len(english_chars) > len(non_english_chars)

def deduplicate_prompts(prompts):
    print(f"Total {len(prompts)} prompts")
    prompts = set(prompts)
    print(f"Total {len(prompts)} prompts after deduplication.")
    
    prompts = list(prompts)
    non_dup = []
    for i in tqdm(range(len(prompts))):
        has_dup = False
        for j in range(i+1, len(prompts)):
            if abs(len(prompts[i]) - len(prompts[j])) > min(len(prompts[i]) * 0.05, len(prompts[j]) * 0.05):
                continue
            if rapidfuzz.fuzz.ratio(prompts[i], prompts[j]) > 95:
                has_dup = True
                break
        if not has_dup:
            non_dup.append(prompts[i])
    prompts = non_dup
    print(f"Total {len(prompts)} prompts after deep deduplication.")
    
    return prompts

def merge_and_deduplicate_prompts(files, out_file):
    """Merge prompts from GPT stores and deduplicates."""
    prompts = []
    for f in files:
        _prompts = torch.load(os.path.join(data_dir, f)) # type: 'list[str]'
        prompts.extend([s.lstrip().rstrip() for s in _prompts])
    
    prompts = deduplicate_prompts(prompts)
    
    fname = os.path.join(data_dir, out_file)
    torch.save(prompts, fname)
    print(f">> Save merged GPTs prompts to {fname}")

if __name__ == "__main__":
    # extract_leaked_GPTs()  # start with "You are a ChatGPT."
    # extract_leaked_GPTs_no_pattern()
    # extract_general_prompts()
    
    # merge_and_deduplicate_prompts(files=['GPTs.pth', 'Leaked-GPTs.pth'], out_file='merged_GPTs.pth')
    
    # merge_and_deduplicate_prompts(files=['BlackFriday-GPTs-Prompts.pth'], out_file='dedup_BlackFriday-GPTs-Prompts.pth')
    
    categories = ['Academic', 'Business', 'Creative', 'Game', 'Job-Hunting', 'Marketing', 'Productivity-&-life-style', 'Programming']
    for cat in categories:
        print(f"== {cat} ==")
        merge_and_deduplicate_prompts(files=[os.path.join('blackfriday', f"{cat}.pth")], out_file=os.path.join('blackfriday', f"dedup_{cat}.pth"))
