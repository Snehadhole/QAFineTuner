from transformers import GenerationConfig
from ModelTokenizer import get_model_tokenizer

def ModelInference(test_text,model_path) :
    trained_model, tokenizer = get_model_tokenizer(model_path)

    tokenized_test_text = tokenizer(test_text,
                                  return_tensors='pt')
    model_output = trained_model.generate(tokenized_test_text.input_ids,
                                          generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),)[0]
    final_output = tokenizer.decode(model_output, skip_special_tokens=True)
    return final_output
