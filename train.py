from datasets import load_dataset
from colorama import Fore
from dotenv import load_dotenv
import os
import multiprocessing

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def format_chat_template(batch, tokenizer):
    system_prompt = "You help extract information from rental documents."
    samples = []

    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + "
            "'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples
    }

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA drivers "
            "before running this fine-tuning script."
        )

    print(Fore.LIGHTGREEN_EX + f"CUDA available: {torch.cuda.is_available()}" + Fore.RESET)
    print(Fore.LIGHTGREEN_EX + f"GPU: {torch.cuda.get_device_name(0)}" + Fore.RESET)

    # Load dataset from JSON file
    dataset = load_dataset("json", data_files="data/data.json", split="train")
    print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

    base_model = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        num_proc=1,  # keep 1 for Windows safety
        batched=True,
        batch_size=1
    )
    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    device_map="auto",
    quantization_config=quant_config,
    token=HF_TOKEN,
    cache_dir="./model",
    )


    print(Fore.CYAN + str(model) + Fore.RESET)
    print(Fore.LIGHTYELLOW_EX + str(next(model.parameters()).device))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir="finetuned",
            num_train_epochs=20,
            bf16=False,
            per_device_train_batch_size=1,
            fp16=False
        ),
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("complete_checkpoint1")
    trainer.model.save_pretrained("final_model1")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()