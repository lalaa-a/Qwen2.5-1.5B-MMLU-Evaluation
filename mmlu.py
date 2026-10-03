import torch
from datasets import  load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ds = load_dataset(
    "cais/mmlu", "high_school_geography",split="test"
).select(range(100))

print(ds[0])

#importing model

name  = "Qwen/Qwen2.5-1.5B"
tok   = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")

letter_ids = [tok.encode("" + l, add_special_tokens=False)[-1] for l in "ABCD"]

def predict(q, choices):
  prompt = "answer the multiple choice question \n\n{q}\n"
  for l,c in zip("ABCD", choices):
    prompt += f"{l} . {c}\n"
  prompt += "Answer:"
  inputs = tok(prompt, return_tensors="pt").to(model.device)
  with torch.no_grad():
    logits = model(**inputs).logits[0, -1]
  return logits[letter_ids].argmax().item()

correct = sum(predict(r["question"], r["choices"]) == r["answer"] for r in ds)
print(f"Accuracy: {correct}/100")

