import dashscope
from http import HTTPStatus
from tqdm import tqdm
import json

# API configuration
dashscope.api_key = "sk-ead2b1****" # 请替换为您的API密钥
MODEL_NAME = "qwen-plus"
output_file = 'amr_clauses.txt'
input_amr_file = '../result/virtual_test.with_r.with_extra.relation.literal.sync_with_no_r.with_func_words.camr_tuple'
input_test_file = '../../data/test_data.json'

def chat(messages):
    response = dashscope.Generation.call(
        model=MODEL_NAME,
        messages=messages,
        result_format='message'
    )
    
    if response.status_code == HTTPStatus.OK:
        messages.append({
            'role': response.output.choices[0]['message']['role'],
            'content': response.output.choices[0]['message']['content']
        })
    else:
        print(f'Request id: {response.request_id}, Status code: {response.status_code}, '
              f'error code: {response.code}, error message: {response.message}')
        messages = messages[:-1]
    return response, messages

def read_tuples(file_path):
    """Read and group AMR triples by sentence ID"""
    sentence_groups = {}
    current_id = None
    current_triples = []
    header_skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过前两行表头
            if header_skipped < 2:
                header_skipped += 1
                continue
                
            if not line:
                if current_id and current_triples:
                    sentence_groups[current_id] = current_triples
                current_triples = []
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                sent_id = parts[0]
                # 确保sent_id是数字
                if not sent_id.isdigit():
                    continue
                    
                if current_id != sent_id:
                    if current_id and current_triples:
                        sentence_groups[current_id] = current_triples
                    current_id = sent_id
                    current_triples = []
                current_triples.append(parts)
    
    if current_id and current_triples:
        sentence_groups[current_id] = current_triples
        
    return sentence_groups

def convert_triples_to_clauses():
    # 读取所有三元组数据
    sentence_groups = read_tuples(input_amr_file)
    
    # 读取测试数据
    with open(input_test_file, 'r', encoding='utf-8') as file:
        data = file.read()
    data = eval(data)

    # 建立句子ID到选项的映射
    choices = ["A", "B", "C", "D"]
    sentence_to_option = {}
    current_sent_id = 1303
    for item in data:
        for qa in item['qas']:
            for i, option in enumerate(qa['options']):
                sentence_to_option[str(current_sent_id + i)] = str(item['cid']) + '_' + str(choices[i])+ '_' + str(current_sent_id + i)
            current_sent_id += 4

    results = []
    
    # 处理所有句子的三元组
    for sent_id, triples in sentence_groups.items():
        # Format triples for the model
        triples_text = "\n".join(["\t".join(triple) for triple in triples])
        
        # Prompt
        prompt = (
                 f"请将以下AMR三元组转换成通顺的中文子句。这些三元组来自同一个句子，需要将它们组合成有意义的子句数组。\n\n"
                 f"转换规则：\n"
                 f"1. :arg0表示动作的执行者，:arg1表示动作的接受者\n"
                 f"2. :time表示时间信息\n"
                 f"3. :manner表示动作的方式\n"
                 f"4. :mod表示修饰关系\n"
                 f"5. :aspect表示动态助词，如'了'、'着'等\n"
                 f"6. :poss表示所属关系\n"
                 f"7. :location表示地点信息\n"
                 f"8. :op1, :op2等表示并列关系\n"
                 f"9. 注意保持概念之间的逻辑关系\n\n"
                 f"三元组：\n{triples_text}\n\n"
                 f"请严格按照以下格式输出，['子句1', '子句2', ...]，不要有解释或其他内容。\n####特别注意：每个子句尽可能保留主语，且不是模糊的指代，例如他、她、它、这个、那个等。不要有解释或其他内容")
        
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        
        response, _ = chat(messages)
        
        if response.status_code == HTTPStatus.OK:
            clauses = response.output.choices[0].message.content
            
            # Self-check
            check_prompt = (f"请检查以下子句数组是否符合要求：每个子句都应该有明确的主语，不能使用模糊的指代词（如'他'、'她'、'它'、'这个'、'那个'等）。\n\n"
                          f"原始子句数组：{clauses}\n\n"
                          f"如果发现有使用模糊指代的情况，请将其替换为具体的主语。表达相同意思的子句只保留一个，避免冗余，严格安装以下格式输出修正后的数组，格式：['子句1', '子句2', ...]，不要做任何解释，不要有任何额外输出。")
            
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': check_prompt}
            ]
            
            check_response, _ = chat(messages)
            
            if check_response.status_code == HTTPStatus.OK:
                clauses = check_response.output.choices[0].message.content
            
            results.append({
                'sent_id': sentence_to_option[str(sent_id)],
                'clauses': clauses
            })
        else:
            results.append({
                'sent_id': sentence_to_option[str(sent_id)],
                'clauses': ''
            })
        print(sentence_to_option[str(sent_id)],clauses)
                

    print("\n最终结果:")
    for result in results:
        print(f"Sentence {result['sent_id']}: {result['clauses']}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['sent_id']}\t{result['clauses']}\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    convert_triples_to_clauses()