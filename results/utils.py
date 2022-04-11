from tqdm import tqdm
import os

def get_arg(textline, field):
    return eval(textline[textline.index(field+'='):textline.find(',',textline.index(field+'='),-1)].split('=')[1])

def extract_results(log_root_path, customized_args = [], file_name_identifier = "train_and_eval"):
    result_dict = {}
    for j,file in tqdm(enumerate(os.listdir(log_root_path))):
        if file_name_identifier in file:
            print(file)
            args = None
            model_name = ""
            results = []
            found = 0
            with open(os.path.join(log_root_path, file), 'r') as fin:
                for i,line in enumerate(fin):
                    if i == 0:
                        model_name = get_arg(line, 'model')
                    if i == 1:
                        args = line.strip()[10:-1]
                    elif "Test set performance" in line:
                        found = 2
                    elif found > 0:
                        found -= 1
                        if found == 0:
                            results.append(eval(line))
            if len(results) > 0:
                args += ','
                result_dict[j] = {'args': args}
                result_dict[j]['model_name'] = model_name
                for k in customized_args:
                    try:
                        result_dict[j][k] = get_arg(args, k)
                    except:
                        result_dict[j][k] = 'NaN'
                results = {k:[result[k] for result in results] for k in results[0].keys()}
                for k,v in results.items():
                    result_dict[j][k] = v
    return result_dict

def extract_transfer_results(log_root_path, customized_args = [], file_name_identifier = "train_and_eval"):
    result_dict = {}
    for j,file in tqdm(enumerate(os.listdir(log_root_path))):
        if file_name_identifier in file:
            print(file)
            args = None
            model_name = ""
            results = []
            found = 0
            with open(os.path.join(log_root_path, file), 'r') as fin:
                for i,line in enumerate(fin):
                    if i == 0:
                        model_name = get_arg(line, 'model')
                    if i == 1:
                        args = line.strip()[10:-1]
                    elif "Test set performance" in line:
                        found = 2
                    elif found > 0:
                        if "Result dict:" in line:
                            found = 1
                        elif found == 1:
                            found == 0
                            results.append(eval(line))
            if len(results) > 0:
                args += ','
                result_dict[j] = {'args': args}
                result_dict[j]['model_name'] = model_name
                for k in customized_args:
                    try:
                        result_dict[j][k] = get_arg(args, k)
                    except:
                        result_dict[j][k] = 'NaN'
                results = {k:[result[k] for result in results] for k in results[0].keys()}
                for k,v in results.items():
                    result_dict[j][k] = v
    return result_dict