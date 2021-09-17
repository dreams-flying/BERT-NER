#! -*- coding:utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from keras.models import Model
from tqdm import tqdm
from bert4keras.layers import *
import extract_chinese_and_punct
chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()

keras.utils.get_custom_objects().update(custom_objects)

maxlen = 128

num_predicate = 10  #实际类别数
num_labels = 2*num_predicate+2
cls_label_1 = num_predicate+1
cls_label_2 = num_predicate

valid_data_path = 'data/cluener/raw_data/dev.json'
label_map_path = 'data/cluener/label_map.json'
entity_label_map_path = 'data/cluener/entity_label_map.json'

config_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

def load_data_v1(filename):
    """
    加载cluener和GENIA
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            tmp = []
            for label_entity in line['label']:
                for k, v in line['label'][label_entity].items():
                    tmp.append((label_entity, k, v[0][0], v[0][1]))
            D.append(
                {
                    "text": line['text'],
                    "entity": tmp
                }
            )

    return D

def load_data_v2(filename):
    """
    加载zh_msra
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            tmp = []
            for label_entity in line['label']:
                for k, v in label_entity.items():
                    tmp.append((k, v))
            D.append(
                {
                    "text": line['text'],
                    "entity": tmp
                }
            )

    return D

# 加载数据集
valid_data = load_data_v1(valid_data_path)
print('valid_data: ', len(valid_data))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 补充输入
labels = Input(shape=(None, num_labels), name='Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model = "albert", #预训练模型选择albert时开启
    return_keras_model=False,
)

# 预测
output = Dense(units=num_labels, kernel_initializer=bert.initializer
)(bert.model.output)#, activation='sigmoid'

entity_model = Model(bert.model.inputs, output)

# 训练模型
train_model = Model(bert.model.inputs + [labels], output)
# train_model.summary()

def start_end_index(text):
    sub_text = []
    buff = ""
    for char in text:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)[1:-1]
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= maxlen - 2:
                break
        else:
            continue
        break
    return tok_to_orig_start_index, tok_to_orig_end_index

def post_process(inference):
    # this post process only brings limited improvements (less than 0.5 f1) in order to keep simplicity
    # to obtain better results, CRF is recommended
    reference = []
    for token in inference:
        token_ = token.copy()
        token_[token_ >= 0.5] = 1
        token_[token_ < 0.5] = 0
        reference.append(np.argwhere(token_ == 1))

    #  token was classified into conflict situation (both 'I' and 'B' tag)
    for i, token in enumerate(reference[:-1]):
        if [0] in token and len(token) >= 2:
            if [1] in reference[i + 1]:
                inference[i][0] = 0
            else:
                inference[i][2:] = 0

    #  token wasn't assigned any cls ('B', 'I', 'O' tag all zero)
    for i, token in enumerate(reference[:-1]):
        if len(token) == 0:
            if [1] in reference[i - 1] and [1] in reference[i + 1]:
                inference[i][1] = 1
            elif [1] in reference[i + 1]:
                inference[i][np.argmax(inference[i, 1:]) + 1] = 1

    #  handle with empty spo: to be implemented

    return inference

def format_output(example, predict_result, entity_label_map,
                  tok_to_orig_start_index, tok_to_orig_end_index):
    # format prediction into example-style output
    predict_result = predict_result[1:len(predict_result) - 1]  # remove [CLS] and [SEP]
    text_raw = example

    flatten_predict = []
    for layer_1 in predict_result:
        for layer_2 in layer_1:
            flatten_predict.append(layer_2[0])

    entity_id_list = []
    for cls_label in list(set(flatten_predict)):
        if 1 < cls_label <= cls_label_1:
            entity_id_list.append(cls_label)
    #
    entity_id_list = list(set(entity_id_list))
    def find_entity(id_, predict_result):
        entity_list = []
        for i in range(len(predict_result)):  # i为当前的判断位置
            if [id_] in predict_result[i]:
                j = 0
                while i + j + 1 < len(predict_result):
                    if [id_ + num_predicate] in predict_result[i + j + 1]:  # 找尾字
                        j += 1
                        break
                    if [1] in predict_result[i + j + 1]:
                        j += 1
                    elif (i + j + 2) < len(predict_result) and [1] in predict_result[i + j + 2]:
                        j += 1
                    else:
                        break
                entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                          tok_to_orig_end_index[i + j] + 1])
                entity_list.append((entity, i, i+j))
        return entity_list

    entity_label_list = []
    for id_ in entity_id_list:
        entitys = find_entity(id_, predict_result)
        for entity_ in entitys:
            entity_label_list.append({
                "entity": entity_[0],
                "label": entity_label_map['label'][id_],
                'start': entity_[1],
                'end': entity_[2]
            })
    return entity_label_list

with open(entity_label_map_path) as f:
    for line in f:
        entity_label_map = json.loads(line)

def extract_entity(text):
    """抽取输入text所包含的实体
    """
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    # 抽取entity
    entity_preds = entity_model.predict([[token_ids], [segment_ids]])
    entity_preds = entity_preds[0, :, :]

    # some simple post process
    # entity_preds = post_process(entity_preds)

    # logits -> classification results
    entity_preds[entity_preds > 0.0] = 1
    entity_preds[entity_preds <= 0.0] = 0

    tok_to_orig_start_index, tok_to_orig_end_index = start_end_index(text)

    predict_result = []
    for token in entity_preds:
        predict_result.append(np.argwhere(token == 1).tolist())

    #calculate metric
    formated_result = format_output(
        text, predict_result, entity_label_map,
        tok_to_orig_start_index, tok_to_orig_end_index)

    return formated_result

def evaluate_v1(data):
    """评估函数，计算f1、precision、recall
    评估cluener和GENIA
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        y_pred = extract_entity(d['text'])
        entity_label_pred = []
        entity_label_y = []
        for entity_label in y_pred:
            entity_label_pred.append((entity_label["label"], entity_label["entity"], entity_label["start"], entity_label["end"]))
        y = d['entity']
        for entity_label in y:
            entity_label_y.append((entity_label[0], entity_label[1], entity_label[2], entity_label[3]))

        R = set([Entity(entity) for entity in entity_label_pred])
        T = set([Entity(entity) for entity in entity_label_y])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))

    pbar.close()
    return f1, precision, recall

def evaluate_v2(data):
    """评估函数，计算f1、precision、recall
    评估zh_msra
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        y_pred = extract_entity(d['text'])
        entity_label_pred = []
        entity_label_y = []
        for entity_label in y_pred:
            entity_label_pred.append((entity_label["label"], entity_label["entity"]))
        y = d['entity']
        for entity_label in y:
            entity_label_y.append((entity_label[0], entity_label[1]))

        R = set([Entity(entity) for entity in entity_label_pred])
        T = set([Entity(entity) for entity in entity_label_y])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))

    pbar.close()
    return f1, precision, recall

class Entity(tuple):
    """用来存实体的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个实体是否等价时容错性更好。
    """
    def __init__(self, entity):
        self.entityx = (
            tuple(tokenizer.tokenize(entity[0])),
            entity[1],
        )

    def __hash__(self):
        return self.entityx.__hash__()

    def __eq__(self, entity):
        return self.entityx == entity.entityx

class NpEncoder(json.JSONEncoder):
    #解决TypeError: Object of type 'int64' is not JSON serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def predict_to_file():
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    in_file = './data/cluener/raw_data/test.json'
    out_file = './cluener_test.json'
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            tempDict = {}
            l = json.loads(l)
            tempDict["id"] = l['id']
            tempDict["label"] = {}
            R = extract_entity(l['text'])
            for d in R:
                if d['label'] not in tempDict["label"]:
                    tempDict['label'][d['label']] = {}
                if d['entity'] not in tempDict['label'][d['label']]:
                    tempDict['label'][d['label']][d['entity']] = []
                tempDict['label'][d['label']][d['entity']].append([d['start'], d['end']])
            l1 = json.dumps(tempDict, cls=NpEncoder, ensure_ascii=False)
            fw.write(l1 + '\n')
    fw.close()

if __name__ == '__main__':
    # 加载模型
    train_model.load_weights('./save/79700/best_model.weights')

    # 测试
    text1 = "在审查时也是有所限制的，改编于好莱坞版惊险片《一线声机》的、陈木胜导演的《保持通话》便体现了这点，"
    text2 = "中共中央致中国致公党十一大的贺词"
    R1 = extract_entity(text2)
    print(R1)

    # 评估数据
    f1, precision, recall = evaluate_v1(valid_data)
    print('f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))

    # predict_to_file()