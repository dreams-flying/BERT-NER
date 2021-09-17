import pdb
import json
import os

sent_list = []
max_len = 0
num_thresh = 0
count = 0

if not os.path.exists('data'):
    os.makedirs('data')

filename_list = ["raw_data/processed_genia/genia.train", "raw_data/processed_genia/genia.dev", "raw_data/processed_genia/genia.test"]
output_list = ["data/genia_train.json", "data/genia_dev.json", "data/genia_test.json"]

for filename, output in zip(filename_list, output_list):
    fw = open(output, 'w', encoding='utf-8')

    with open(filename) as f:
        for line in f:
            # print(line)
            tempDict = {}
            tempDict['text'] = line.replace('\n', '')
            line = line.strip()
            if line == "":  # last few blank lines
                break
            raw_tokens = line.split(' ')
            tokens = raw_tokens

            entities = next(f).strip()
            # print(entities)
            if entities == "":  # no entities
                count += 1
            else:
                entity_list = []
                entities = entities.split("|")
                tempDict1 = {}
                for item in entities:
                    pointers, label = item.split()
                    pointers = pointers.split(",")
                    if int(pointers[1]) > len(tokens):
                        pdb.set_trace()
                    span_len = int(pointers[1]) - int(pointers[0])
                    assert (span_len > 0)
                    entity_str = ' '.join(tokens[int(pointers[0]):int(pointers[1])])
                    if label not in tempDict1:
                        tempDict1[label] = {}
                        tempDict1[label][entity_str] = [[int(pointers[0]), int(pointers[1])]]
                    else:
                        tempDict1[label][entity_str] = [[int(pointers[0]), int(pointers[1])]]
                tempDict['label'] = tempDict1
                l1 = json.dumps(tempDict, ensure_ascii=False)
                fw.write(l1)
                fw.write('\n')

            assert next(f).strip() == ""  # separating line
