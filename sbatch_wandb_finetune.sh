#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=62g
#SBATCH --output=log/slurm/%j.out                              
#SBATCH --error=log/slurm/%j.out 
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --job-name=pbe
#SBATCH --time=6:00:00 # hh:mm:ss for the job
# #SBATCH --time=0:10:00 # hh:mm:ss for the job
#SBATCH --gres=gpu:h100:1

# Purge the module environment to avoid conflicts
module purge
module load WebProxy
module load Anaconda3/2022.10
module list

# Benchmark info
echo "TIMING - Starting job at: $(date)"

# activate user's environment if environment to activate is not blank
# activate conda environment
source /sw/eb/sw/Anaconda3/2022.10/bin/activate /scratch/user/u.jh123957/.conda/envs/pii-leakage-v1
export PATH=/scratch/user/u.jh123957/.conda/envs/pii-leakage-v1/bin:$PATH

cd /scratch/user/u.jh123957/LLM-PBE//finetune || exit
echo "Job is starting on $(hostname)"
which python3
# which wandb

export WANDB_PROJECT=llm-pbe

# srun "wandb agent --count 1 $@"
wandb agent --count 1 $@

# python fine_tune.py --config_path=configs/fine-tune/$@

# python -c "import transformers"

# #SBATCH --output="logs/results/machine_ethics/${target_model}/${setting}-${few_shot_num}-${test_num}-${jailbreak_prompt}-${evasive_sentence}.out"
#  #SBATCH --error="logs/results/machine_ethics/${target_model}/${setting}-${few_shot_num}-${test_num}-${jailbreak_prompt}-${evasive_sentence}.err"

exit
