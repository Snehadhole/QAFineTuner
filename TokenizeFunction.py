from transformers import AutoTokenizer

with open('config.json', 'r') as file:
  config = json.load(file)

model_name = config['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    start_prompt = '\n\n'
    end_prompt = '\n\nAnswer: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["question"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=200).input_ids
    example['labels'] = tokenizer(example["answer"], padding="max_length", truncation=True, return_tensors="pt",max_length=200).input_ids

    return example
