# QAFineTuner

This repository contains a Python script designed for fine-tuning a Large Language Model (LLM) for question-answering tasks. The code leverages the google/flan-t5-base model from Hugging Face's Transformers library as the base model, allowing for efficient fine-tuning using a pre-trained sequence-to-sequence language model.

## Features:    
**Model Loading**: The script loads the google/flan-t5-base model using AutoModelForSeq2SeqLM from Hugging Face, ensuring that the model is ready for fine-tuning on custom datasets.    
**Tokenizer Initialization**: Along with the model, the associated tokenizer is also loaded using AutoTokenizer, which is crucial for preparing text data for training and inference.    
**Torch Compatibility**: The model is loaded with torch_dtype=torch.float32 to optimize performance during training.

# Installation
###Clone the Repository    
```  
git clone https://github.com/Snehadhole/QAFineTuner.git
```
###Navigate to the Directory:
```
cd QAFineTuner
```

To install the required Python packages, use the `requirements.txt` file provided in the repository. Run the following command:    
```
pip install -r requirements.txt
```
###Run the Script
```
python qa_fine_tuner.py
```

## Example Prediction Script

Here’s an example script for predicting answers using a pre-trained model. This script loads configuration from a JSON file, initializes a model, and makes a prediction based on the input text.

```python
import json
from transformers import GenerationConfig

# Example text to be predicted
test_text = "What is CUDA Nsight?"

# Load configuration from JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

# Get model path from configuration
model_path = config["model_path"]

# Initialize ModelInference and get prediction
result = ModelInference(test_text, model_path)

# Print the result
print(result)

