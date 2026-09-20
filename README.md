# Rental Document Information Extraction - Fine-Tuning Pipeline

A fine-tuning pipeline for training Meta Llama 3.2 1B to extract information from rental documents using QLoRA, PEFT, and TRL SFTTrainer.

# Video Explanation:https://youtu.be/3XD_2jkCtaE

## Features

- Fine-tunes Meta Llama 3.2 1B for rental document information extraction.
- Uses QLoRA with 4-bit quantization.
- Conversational chat-template training format.
- PEFT LoRA adapters.
- TRL SFTTrainer.
- Gradient checkpointing support.

## Project Structure

├── train.py
├── data/
│   └── data.json
├── model/
├── finetuned/
├── complete_checkpoint1/
├── final_model1/
└── .env

## Requirements

Python 3.10+

Install dependencies from requirements.txt

## Environment Variables

Create a .env file:

HF_TOKEN=your_huggingface_token

## Dataset Format

Example:

{
  "question": "What is the monthly rent?",
  "answer": "$1500"
}

## Base Model

meta-llama/Llama-3.2-1B


## Training

Run:

python train.py

The script:
1. Checks CUDA availability.
2. Loads dataset.
3. Formats training samples.
4. Loads model in 4-bit mode.
5. Applies LoRA.
6. Trains the model.
7. Saves checkpoints.

## Output

- finetuned/
- complete_checkpoint1/
- final_model1/

## Hardware Requirements

- NVIDIA GPU
- CUDA-enabled PyTorch
- Recommended: 4GB+ VRAM

## Use Cases

- Lease agreement analysis
- Rental contract information extraction
- Property management automation
- Document question answering
