
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def respond_to_mood(mood):
    prompt = f"<s>[INST] The user says they feel {mood}. Respond with a short, kind, supportive message. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

iface = gr.Interface(fn=respond_to_mood,
                     inputs="text",
                     outputs="text",
                     title="MoodMate (Mistral Edition)",
                     description="Tell MoodMate how you're feeling 💬")

iface.launch()