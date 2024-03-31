import together
import os

save_dir = "checkpoints/llama2-echr"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

together.Finetune.download(
    fine_tune_id="ft-f69d6007-5c71-4638-b168-14fe607040b5",
    output = f"{save_dir}/model.tar.zst"
)
