# import json
# import jieba
# from tqdm import tqdm

# with open('../../data/acrc_test.json', 'r', encoding='utf-8') as file:
#     data = file.read()
# data = eval(data)['data']

# total_options = len(data) * 4
# option_id = 1

# with open('acrc_test.txt', 'w', encoding='utf-8') as f1:
#     with open('acrc_test_with_id.txt', 'w', encoding='utf-8') as f2:
#         with tqdm(total=total_options, desc="Processing") as pbar:
#             for item in data:
#                 for qa in item['qas']:
#                     for option in range(4):
#                         words = jieba.cut(qa['options'][option])
#                         line = ' '.join(words)
#                         f1.write(f'{line}\n')
#                         f2.write(f'{option_id}\t{line}\n')
#                         option_id += 1
#                         pbar.update(1)

import json
import jieba
import re

class PATHS:
    test_path = "../../data/test_data.json"
    output_path = './virtual_test.txt'
    output_path_with_id = './virtual_test_with_id.txt'

def get_options(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    options = []
    for item in data:
        for qa in item['qas']:
            options.extend(qa['options'])
    
    return options

def normalize_text(text):
    """规范化文本处理"""
    # 1. 移除多余空格
    text = ' '.join(text.split())
    # 2. 确保标点符号前后有空格
    text = re.sub(r'([，。、？！；：""''（）《》])', r' \1 ', text)
    # 3. 再次移除多余空格
    text = ' '.join(text.split())
    return text

def segment_text(text):
    """分词处理"""
    # 1. 先规范化文本
    text = normalize_text(text)
    # 2. 使用jieba分词
    words = jieba.cut(text)
    # 3. 规范化分词结果
    return ' '.join(words)

def process_options(options):
    return [segment_text(option) for option in options]

def save_output(options, path, with_id=False):
    with open(path, 'w', encoding='utf-8') as f:
        for i, option in enumerate(options, start=1):
            if with_id:
                f.write(f"{i+1302}\t{option}\n")
            else:
                f.write(f"{option}\n")

# 主程序
options = get_options(PATHS.test_path)
segmented_options = process_options(options)

# 保存前进行最后的检查
def validate_segmentation(text):
    """验证分词结果是否规范"""
    # 检查是否存在连续空格
    if '  ' in text:
        return False
    # 检查开头和结尾是否有空格
    if text.startswith(' ') or text.endswith(' '):
        return False
    return True

# 在保存前进行验证
validated_options = []
for option in segmented_options:
    if not validate_segmentation(option):
        # 如果发现不规范的分词，重新规范化
        option = ' '.join(option.split())
    validated_options.append(option)

save_output(validated_options, PATHS.output_path)
save_output(validated_options, PATHS.output_path_with_id, with_id=True)

print("处理完成。输出文件已保存。")