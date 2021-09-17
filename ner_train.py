#! -*- coding:utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import extract_chinese_and_punct
chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()

maxlen = 128
batch_size = 40
epoch = 10

num_predicate = 10  #实际类别数
num_labels = 2*num_predicate+2
cls_label_1 = num_predicate+1
cls_label_2 = num_predicate

#数据路径
train_data_path = 'data/cluener/raw_data/train.json'
valid_data_path = 'data/cluener/raw_data/dev.json'
label_map_path = 'data/cluener/label_map.json'
entity_label_map_path = 'data/cluener/entity_label_map.json'

#模型路径
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
train_data = load_data_v1(train_data_path)
print('train_data:', len(train_data))

valid_data = load_data_v1(valid_data_path)
print('valid_data:', len(valid_data))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

with open(label_map_path) as f:
    for line in f:
        label_map = json.loads(line)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels= [], [], []
        for is_end, d in self.sample(random):
            text_raw = d['text']
            sub_text = []
            buff = ""
            for char in text_raw:
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

            labels = [[0] * num_labels for i in range(len(tokens))]  # initialize tag

            for entity in d["entity"]:
                if entity[0] in label_map.keys():
                    label_subject = label_map[entity[0]]
                    subject_sub_tokens = tokenizer.tokenize(entity[1])[1:-1]
                    for index in range(len(tokens) - len(subject_sub_tokens) + 1):
                        if tokens[index:index + len(subject_sub_tokens)] == subject_sub_tokens:
                            labels[index][label_subject] = 1
                            for i in range(len(subject_sub_tokens) - 1):
                                labels[index + i + 1][1] = 1
                            if len(subject_sub_tokens) > 1:
                                labels[index + len(subject_sub_tokens) - 1][label_subject + num_predicate] = 1
                                labels[index + len(subject_sub_tokens) - 1][1] = 0
                            break

            # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
            for i in range(len(labels)):
                if labels[i] == [0] * num_labels:
                    labels[i][0] = 1

            # add [CLS] and [SEP] token, they are tagged into "O" for outside
            if len(tokens) > maxlen - 2:
                tokens = tokens[0:(maxlen - 2)]
                labels = labels[0:(maxlen - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            outside_label = [[1] + [0] * (num_labels - 1)]
            labels = outside_label + labels + outside_label

            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)

            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=128)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=128)
                batch_labels = sequence_padding(batch_labels, length=128)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 补充输入
labels = Input(shape=(None, num_labels), name='Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model = "albert", #预训练模型选择albert时开启
    return_keras_model=False,
)

# 预测entity
output = Dense(units=num_labels, kernel_initializer=bert.initializer
)(bert.model.output)#, activation='sigmoid'
entity_model = Model(bert.model.inputs, output)

# 训练模型
train_model = Model(
    bert.model.inputs + [labels], output)
# train_model.summary()

input_mask = bert.model.get_layer('Embedding-Token').output_mask
input_mask = K.cast(input_mask, K.floatx())

#第一种损失函数
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

entity_loss = multilabel_categorical_crossentropy(labels, output)
entity_loss = K.sum(entity_loss * input_mask) / K.sum(input_mask)

#第二种损失函数
# entity_loss = K.categorical_crossentropy(labels, output)
# entity_loss = K.sum(entity_loss * input_mask) / K.sum(input_mask)

train_model.add_loss(entity_loss)

# AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = Adam(lr=1e-5)
train_model.compile(optimizer=optimizer)

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
    # calculate metric
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

class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = -0.1
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate_v1(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('./best_model.weights')
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=[evaluator]
    )