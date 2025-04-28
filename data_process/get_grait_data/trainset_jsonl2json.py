import json

with open("dataset/RAIT_dataset/mmlu/llama-3-8b-instruct-hf/Cor_RAIT.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data_van = []
new_data_idk = []
for example in data:
    # 提取每一个问题的相关信息
    _task_name = example['prompt_task_name']
    _question = example['prompt_question']
    _answer = example['prompt_answer']
    _hint = example['prompt_instruction'].format(task_name=_task_name)
    
    # 提取问题和选项
    qid = example['id']
    question = example['question']
    A = example['A']
    B = example['B']
    C = example['C']
    D = example['D']

    answer = example['target']
    
    # 构建完整的提示信息
    prompt = f'{_hint}\n{_question}: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n{_answer}: '
    per_data_van = {
        "dataset": "rehearsal_mmlu",
        "id": qid,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }
    per_data_idk = {
        "dataset": "rehearsal_mmlu",
        "id": qid,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "I don't know."
            }
        ]
    }
    new_data_van.append(per_data_van)
    new_data_idk.append(per_data_idk)

with open("dataset/RAIT_dataset/mmlu/llama-3-8b-instruct-hf/VAN.jsonl", 'w', encoding='utf-8') as file:
    for item in new_data_van:
        # 将每个列表项作为单独的JSON对象写入新的一行
        file.write(json.dumps(item) + '\n')
        
with open("dataset/RAIT_dataset/mmlu/llama-3-8b-instruct-hf/IDK.jsonl", 'w', encoding='utf-8') as file:
    for item in new_data_idk:
        # 将每个列表项作为单独的JSON对象写入新的一行
        file.write(json.dumps(item) + '\n')
