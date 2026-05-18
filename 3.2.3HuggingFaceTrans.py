'''下面这段代码会自动从 Hugging Face Hub 下载所需的模型文件和分词器配置'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，如果有GPU则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")