import json
import random

random.seed(42)

with open('/mnt/petrelfs/zhurunchuan/code/LESS/data/train/our/mmlu.json', 'r') as file:
    data = json.load(file)

# 提取 correct_init 和 entropy_init 及其对应的索引
correct_init_values = [(index, item['KQ']['correct_init']) for index, item in enumerate(data)]
entropy_init_values = [(index, item['KQ']['entropy_init']) for index, item in enumerate(data)]

# 提取符合条件的 entropy_init 值及其索引
filtered_correct_values = [
    (index, item['KQ']['correct_init'], item['KQ']['entropy_init']) 
    for index, item in enumerate(data) 
    if item['KQ']['correct_init'] > 0.5
]

filtered_incorrect_values = [
    (index, item['KQ']['correct_init'], item['KQ']['entropy_init']) 
    for index, item in enumerate(data) 
    if item['KQ']['correct_init'] < 0.5
]

sorted_entropy_init = sorted(filtered_correct_values, key=lambda x: x[1], reverse=True)
van_table = [index for index, correct, entropy in sorted_entropy_init[:4000]]
# idk_table_random = random.sample([index for index, correct, entropy in filtered_incorrect_values], 1000)

print("Top 4000 indices for correct_init:", van_table)

van_group = [index for index, correct, entropy in filtered_correct_values]
idk_group = [index for index, correct, entropy in filtered_incorrect_values]

import torch

# 直接加载 .pt 文件
idk_idk_inner = torch.load('grads/llama3-8b-instruct-mmlu-rehearsal/idk_idk_inner.pt')
idk_van_inner = torch.load('grads/llama3-8b-instruct-mmlu-rehearsal/idk_van_inner.pt')
van_van_inner = torch.load('grads/llama3-8b-instruct-mmlu-rehearsal/van_van_inner.pt')

## method6
idk_table = []
idk_sim_dict = {}
for index in idk_group:
    idk_sim_dict[index] = idk_idk_inner[index][idk_group].mean()
idk_table = sorted(idk_sim_dict, key=idk_sim_dict.get, reverse=True)[:1000]

from tqdm import tqdm

idk_table = []
idk_sim_dict = {}
for i in tqdm(range(1000)):
    if i == 0:
        for j in idk_group:
            if j not in idk_table:
                idk_sim_dict[j] = idk_van_inner[j][van_table].sum() + idk_idk_inner[j][idk_table].sum()
    max_id = max(idk_sim_dict, key=idk_sim_dict.get)
    idk_table.append(max_id)
    del idk_sim_dict[max_id]
    for j in idk_sim_dict.keys():
        idk_sim_dict[j] += idk_idk_inner[j][max_id]

idk_table = sorted(idk_sim_dict, key=idk_sim_dict.get)[:1000]

score_table = {}

for idx in idk_table:

    # calculate score_idk
    idk_influence_vanset_van = idk_van_inner[idx][van_group].mean()
    idk_influence_vanset_idk = idk_idk_inner[idx][van_group].mean()
    idk_influence_idkset_van = idk_van_inner[idx][idk_group].mean()
    idk_influence_idkset_idk = idk_idk_inner[idx][idk_group].mean()

    # van_influnence_vantable_van = van_van_inner[idx][van_table].mean()
    score_table[idx] = [idk_influence_vanset_idk, idk_influence_idkset_idk, data[idx]['KQ']['correct_init'], data[idx]['KQ']['entropy_init']]

alphas = {}
for idx in score_table.keys():
    alphas[idx] = score_table[idx][0].item() - score_table[idx][1].item()

import math

def softmax(alphas):
    # 计算指数
    exp_alphas = {idx: math.exp(alpha / 0.05) for idx, alpha in alphas.items()}
    
    # 计算总和
    sum_exp_alphas = sum(exp_alphas.values()) / 500
    
    # 计算 softmax
    softmax_alphas = {idx: exp_alphas[idx] / sum_exp_alphas for idx in exp_alphas.keys()}
    
    return softmax_alphas

def min_max_alphas(alphas):
    # 将 tensors 转换为浮点数列表
    alphas_float = [alphas[idx].item() for idx in alphas.keys()]
    
    # 计算 min 和 max
    min_alpha = min(alphas_float)
    max_alpha = max(alphas_float)
    
    normalized_alphas = {}
    # 进行归一化
    for idx in alphas.keys():
        normalized_alphas[idx] = ((alphas[idx].item() - min_alpha) / (max_alpha - min_alpha))
    
    return normalized_alphas

def standardize_alphas(alphas):
    # 转换为浮点数列表
    alphas_float = torch.tensor([alpha.item() for alpha in alphas])
    
    # 计算均值和标准差
    mean_alpha = alphas_float.mean()
    std_alpha = alphas_float.std()
    
    # 进行标准化
    standardized_alphas = (alphas_float - mean_alpha) / std_alpha
    
    return [alpha.item() for alpha in standardized_alphas]

softmax_alphas = softmax(alphas)
# standardized_alphas = standardize_alphas(alphas)

import copy

new_data = []
for i in van_table:
    per_data = copy.deepcopy(data[i])
    per_data['alpha'] = 1
    per_data['beta'] = 0
    per_data['index'] = i
    new_data.append(per_data)
for i in idk_table:
    per_data = copy.deepcopy(data[i])
    per_data['target'] = "I don't know."
    per_data['alpha'] = softmax_alphas[i]
    per_data['beta'] = 0
    per_data['index'] = i
    new_data.append(per_data)

import copy

new_data = []
for i in van_table:
    per_data = copy.deepcopy(data[i])
    per_data['alpha'] = 1
    per_data['beta'] = 0
    per_data['index'] = i
    new_data.append(per_data)
for i in idk_table:
    per_data = copy.deepcopy(data[i])
    per_data['target'] = "I don't know."
    per_data['alpha'] = 1
    per_data['beta'] = 0
    per_data['index'] = i
    new_data.append(per_data)

random.seed(42)
random.shuffle(new_data)

with open('dataset/RAIT_dataset/mmlu/llama-3-8b-instruct-hf/grait.json', 'w') as json_file:
    json.dump(new_data, json_file, indent=4)