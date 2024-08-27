from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name = "google/flan-t5-base")

def tokenize_function(example):
    start_prompt = '\n\n'
    end_prompt = '\n\nAnswer: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["question"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=200).input_ids
    example['labels'] = tokenizer(example["answer"], padding="max_length", truncation=True, return_tensors="pt",max_length=200).input_ids

    return example
