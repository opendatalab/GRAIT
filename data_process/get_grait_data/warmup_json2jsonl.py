import json

with open("dataset/rehearsal_dataset/mmlu/llama-3-8b-instruct-hf/vanilla_cor0.99_n1000.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []
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
    per_data = {
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
    new_data.append(per_data)

with open("dataset/rehearsal_dataset/mmlu/llama-3-8b-instruct-hf/vanilla_cor0.99_n1000.jsonl", 'w', encoding='utf-8') as file:
    for item in new_data:
        # 将每个列表项作为单独的JSON对象写入新的一行
        file.write(json.dumps(item) + '\n')