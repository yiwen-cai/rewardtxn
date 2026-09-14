import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading base model Qwen2.5-1.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("/root/models/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    "/root/models/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)
model.eval()

# Test prompt
messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.pad_token_id)

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"\nBase model test:")
print(f"Prompt: What is 2+2?")
print(f"Response: [{response}]")
print(f"Expected: 4")
