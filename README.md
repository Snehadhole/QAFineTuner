# QAFineTuner

This repository contains a Python script designed for fine-tuning a Large Language Model (LLM) for question-answering tasks. The code leverages the google/flan-t5-base model from Hugging Face's Transformers library as the base model, allowing for efficient fine-tuning using a pre-trained sequence-to-sequence language model.

## Features:    
**Model Loading**: The script loads the google/flan-t5-base model using AutoModelForSeq2SeqLM from Hugging Face, ensuring that the model is ready for fine-tuning on custom datasets.    
**Tokenizer Initialization**: Along with the model, the associated tokenizer is also loaded using AutoTokenizer, which is crucial for preparing text data for training and inference.    
**Torch Compatibility**: The model is loaded with torch_dtype=torch.float32 to optimize performance during training.

# Installation
To install the required Python packages, use the `requirements.txt` file provided in the repository. Run the following command:    
```bash
pip install -r requirements.txt

1. # Clone the Repository
'''bash
git clone https://github.com/Snehadhole/QAFineTuner.git
cd QAFineTuner
