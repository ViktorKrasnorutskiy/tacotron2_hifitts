import os
import json

dataset_path = '../'

datasets = {}
readers_set = set()

def parse_json(json_data):
    parsed_data = []
    for row in json_data:
        row_data = {}
        for param_name in row:
            row_data[param_name] = row[param_name]
        parsed_data.append(row_data)
    return parsed_data

def read_json(json_path):
    with open(json_path, encoding='utf-8') as f:
        json_data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
        parsed_data = parse_json(json_data)
    return parsed_data


files_list = os.listdir(dataset_path)
for file_name in files_list:
    if 'manifest' not in file_name:
        continue
    dataset_type = file_name.split('_')[3].split('.')[0]
    reader_id = file_name.split('_')[0]
    data_quality = file_name.split('_')[2]
    reader_data = read_json(dataset_path + file_name)
    readers_set.add(reader_id)
    if dataset_type not in datasets:
        datasets[dataset_type] = []
    datasets[dataset_type].append({
        'RI': reader_id,
        'DQ': data_quality,
        'DATA': reader_data
    })

n_readers = len(readers_set)
readers_dict = {ri:i for i,ri in enumerate(readers_set)}


import shutil
#n_files = 16
if 'audio' not in os.listdir():
    os.mkdir('audio')

def create_file(fname, dataset, dataset_type, ri='92'):
    count = 0
    with open(fname, 'w') as f:
        for data in dataset:
            reader_id = data['RI']
            if reader_id != ri:
                continue
            RI = readers_dict[reader_id]
            DQ = data['DQ']
            DATA = data['DATA']
            for d in DATA:

                count += 1
            #    if count > n_files:
            #        break

                from_path = dataset_path + d['audio_filepath']
                FPATH = 'audio/'+str(count)+'_'+dataset_type+'_'+reader_id+'.flac'
                TEXT = d['text']
                line = f'{FPATH}|{TEXT}|{RI}\n'
                f.write(line)

                to_path = './'+FPATH
                shutil.copy(from_path, to_path)

ri = '92'
for dataset_type in datasets:
    dataset = datasets[dataset_type]
    fname = './hifitts_' + dataset_type + '.txt'
    create_file(fname, dataset, dataset_type, ri)
