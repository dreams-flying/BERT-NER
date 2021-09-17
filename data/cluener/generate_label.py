import json

train_data_path = 'raw_data/train.json'
dev_data_path = 'raw_data/dev.json'

fw1 = open('label_map.json', 'w', encoding='utf-8')
fw2 = open('entity_label_map.json', 'w', encoding='utf-8')

label_map = {}
label_map['O'] = 0
label_map['I'] = 1
tempList = []
tempList.append('empty')
tempList.append('empty')
with open(train_data_path, encoding='utf-8') as f1, open(dev_data_path, encoding='utf-8') as f2:
    for line in f1:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label_map:
                label_map[label_entity] = len(label_map)
                tempList.append(label_entity)
    for line in f2:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label_map:
                label_map[label_entity] = len(label_map)
                tempList.append(label_entity)
entity_label_map = {}
entity_label_map['label'] = tempList
l1 = json.dumps(label_map, ensure_ascii=False)
l2 = json.dumps(entity_label_map, ensure_ascii=False)
fw1.write(l1)
fw2.write(l2)