python -m attacks.DataExtraction.extract_enron_local --model EleutherAI/pythia-14m  --arch EleutherAI/pythia-14m --peft none



python -m attacks.DataExtraction.extract_enron_local --num_sample 5 --model LLM-PBE/together-llama-2-7B-enron-scrubbed:checkpoint_1764 --arch meta-llama/Llama-2-7b-hf --peft none


python -m attacks.DataExtraction.extract_enron_local --num_sample 5 --model meta-llama/Llama-2-7b-hf --arch meta-llama/Llama-2-7b-hf --peft none

python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model meta-llama/Llama-2-7b-hf --arch meta-llama/Llama-2-7b-hf --peft none

python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-dp8 --arch meta-llama/Llama-2-7b-hf --peft lora

python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-scrubbed --arch meta-llama/Llama-2-7b-hf --peft lora
python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-undefended --arch meta-llama/Llama-2-7b-hf --peft lora





python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-chat-dp8 --arch meta-llama/Llama-2-7b-hf --peft lora


python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-chat-scrubbed --arch meta-llama/Llama-2-7b-hf --peft lora


python -m attacks.DataExtraction.extract_enron_local --num_sample -1 --model LLM-PBE/enron-llama2-7b-chat-undefended --arch meta-llama/Llama-2-7b-hf --peft lora





