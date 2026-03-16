import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading ARAKKHA...')
tokenizer = AutoTokenizer.from_pretrained('./arakkha-model')
model = AutoModelForCausalLM.from_pretrained('./arakkha-model', torch_dtype=torch.float16, device_map='auto')
model.gradient_checkpointing_disable()
model.config.use_cache = True

messages = [
    {'role': 'system', 'content': 'You are Nyan Lin Kyaw — a CAD freelancer and factory HR manager in Rayong, Thailand. Be direct and decisive.'},
    {'role': 'user', 'content': 'A client offers 80 USD for a SolidWorks assembly job. 2 day deadline. What do you do?'}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('ARAKKHA:', response)
