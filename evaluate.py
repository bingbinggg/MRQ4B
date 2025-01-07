import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(42) 

# 加载 Hugging Face 上的预训练模型和分词器
model_name = "/data1/public/models/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:6")

# 读取数据集
data_path = 'qa.jsonl'
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line.strip()) for line in f]

# 得分映射
score_map = {"low": 1, "medium": 2, "high": 3, "random": 0}

def evaluate_sample(sample):
    question = sample['question']
    options = sample['options']
    # 随机化选项设置
    random.shuffle(options)
    # 构造模型输入的prompt
    # prompt = "下面的问题来自你的女友，请选择最合适的选项作为答案，仅输出选项编号（A/B/C/D）：\n\n"
    # prompt = "你正在面对女友的提问，请为下面的问题选择最合适的选项作为答案，仅输出选项编号（A/B/C/D）：\n\n"
    prompt = "你正在面对女友的提问，请为下面的‘死亡’问题选择‘求生欲’最强的的选项作为答案，仅输出选项编号（A/B/C/D）：\n\n"
    prompt += f"问题：{question}\n选项：\n"
    for i in range(len(options)):
        prompt += f"{chr(65 + i)}. {options[i]['text']}\n"
    prompt += "答案："
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, 
                            max_new_tokens=10, 
                            temperature=1.0,
                            num_beams=4)
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    score = 0
    if response[0] in "ABCD":
        score = score_map.get(options[ord(response[0]) - ord('A')]['label'])
    return score, 1 if score > 0 else 0


if __name__ == "__main__":
    content_score = 0
    format_score = 0
    for sample in dataset:
        content_score1, format_score1 = evaluate_sample(sample)
        content_score += content_score1
        format_score += format_score1
    print(f"score: {format_score/len(dataset)},{content_score/format_score}")
