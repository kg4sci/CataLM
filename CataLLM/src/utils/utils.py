import math
import openai
import paramiko
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer,AutoModel
import torch.nn.functional as F
import sys
sys.path.append('..')
from torch import Tensor
import transformers
import torch

model_name = '/home/wangld/Catalyst_new/model/bert-base-uncased'
# model_name = r'D:\PycharmProjects\Catalyst_V1\model\bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_load = AutoModel.from_pretrained(model_name)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
def encode_with_small_model(text_string):
    batch_dict = tokenizer(text_string, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model_load(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings.tolist()[0]

def embedding_compare(title_embedding, embedding_data, reference_num):
    title_embedding = np.array(title_embedding)
    embedding_title = np.array([i[1] for i in embedding_data])
    title_score = cosine_similarity([title_embedding],embedding_title)
    similar_doi = [embedding_data[i][0] for i in np.argsort(title_score)[0][:reference_num]]
    return similar_doi

def embedding_compare2(title, entity_data, reference_num):
    title_score = [cosine_similarity_str(title, i[1]) for i in entity_data]
    print(title)
    similar_doi = [entity_data[i][0] for i in np.argsort(title_score)[:reference_num]]
    return similar_doi

def context_search(product, material_type, control_method_type, entity_data):
    # print(entity_data)
    context_all = [i for i in entity_data if i[1][0] == product and i[1][1] == material_type.split('means')[1].lstrip() and i[1][2] == control_method_type]
    print(context_all)
    context = ""
    example_num = 1
    for context_i in context_all:
        similar_abstract = context_i[3]
        context += "Reference" + str(example_num) + ":\n"
        context += similar_abstract + "\n\n"
        example_num += 1
    return context

openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://10.0.82.212:8000/v1"
# openai.api_base = "http://10.0.87.17:58001/v1"

model = "llama"


def cosine_similarity_str(s1, s2):
    """
    计算两个字符串的余弦相似度
    """
    def word_frequency(name):
        # 将字符串中的单词按照出现次数降序排列
        # 返回一个字典，键为单词，值为出现次数
        words = name.split()
        freq_dict = dict()
        for word in words:
            freq_dict[word] = freq_dict.get(word,0) + 1
        return freq_dict

    # 计算词频
    freq1 = word_frequency(s1)
    freq2 = word_frequency(s2)
    # 构建词频向量
    intersection = set(freq1.keys()) & set(freq2.keys())
    numerator = sum([freq1[x] * freq2[x] for x in intersection])

    sum1 = sum([pow(freq1[x], 2) for x in freq1.keys()])
    sum2 = sum([pow(freq2[x], 2) for x in freq2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def answer_query(prompt):

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = completion.choices[0].message.content
    return answer


def embedding_query(text_string):
    embedding = openai.Embedding.create(input=text_string, model=model)['data'][0]['embedding']
    return embedding


# def embedding_compare(title_embedding, abstract_embedding, embedding_data, reference_num):
#     title_embedding = np.array(title_embedding)
#     abstract_embedding = np.array(abstract_embedding)
#     embedding_title = np.array([i[1] for i in embedding_data])
#     embedding_abstract = np.array([i[2] for i in embedding_data])
#     title_score = cosine_similarity([title_embedding],embedding_title)
#     abstract_score = cosine_similarity([abstract_embedding], embedding_abstract)
#     total_score = 0.3*title_score + 0.7*abstract_score
#     similar_doi = [embedding_data[i][0] for i in np.argsort(total_score)[0][:reference_num]]
#     return similar_doi



def load_remote_file(path, file_name):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.0.87.16', 22, 'root', 'cnic.cn')
    sftp_client = client.open_sftp()
    remote_root_path = '/opt/mfs/duyi/cataly/'
    remote_file = sftp_client.open(remote_root_path + path + '/' + file_name, 'r')
    data = json.loads(remote_file.read().decode('utf-8'))
    return data
