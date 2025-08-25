results = []
for i, prompt_text in enumerate(tqdm(all_prompts, desc="Generating responses")):
    response_text = 'C'
    
    example = test_df.loc[i, 'context']
    a = test_df.loc[i, 'question']
    b = 'A:' + test_df.loc[i, 'A'] + '\nB:' + test_df.loc[i, 'B'] + '\nC:' + test_df.loc[i, 'C'] + '\nD:' + test_df.loc[i, 'D']
    c = test_df.loc[i, 'tr_context']
    
    #FIRST ROUND
    rawmessages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    read_text_mes = {'role': 'user', 'content': f"请阅读文言文译文，完成任务，任务我会在接下来的对话中给你，如果你阅读完了，请只回复阅读完毕。\n\n{SS}\n文言文译文:\n{test_df.loc[i, 'or_text']}"}
    rawmessages.append(read_text_mes)
    _, mes = chat(rawmessages)
    
    #SECOND ROUND
    text = test_df.loc[i, 'context']
    tokens = list(tokenizer.tokenize(text))

    punctuation = set(['-', '，', '。', '“', '”', '：', '？','?' '！', '‘', '’', '；'])

    cleaned_tokens = []
    seen_tokens = set()

    for token in tokens:
        if token not in seen_tokens and token not in punctuation:
            cleaned_tokens.append(token)
            seen_tokens.add(token)

    annotation_dict = dict(zip(dict_train['word'], dict_train['annotation']))

    annotations = [f"{token}: {annotation_dict[token]}" for token in cleaned_tokens if token in annotation_dict]
    annotations = '\n--------\n'.join(annotations)

    # NEW ROUND: 选项分句
    option_segmentation_prompt = f"""{SS}请将以下选项分句，注意每一句话都要找到对应的主语，不要使用代词，每个选项分局后的句子放入到数组中：\n{b}
                                    输出示例：\nA: ["子句1", "子句2", ...]\nB: ["子句1", "子句2", ...]\nC: ["子句1", "子句2", ...]\nD: ["子句1", "子句2", ...]"""
    option_seg_mes = {'role': 'user', 'content': option_segmentation_prompt}
    mes.append(option_seg_mes)
    res, mes = chat(mes)
    
    if res.status_code == HTTPStatus.OK:
        segmented_options = res.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            res.request_id, res.status_code, res.code, res.message
        ))
        segmented_options = b  

    # THIRD ROUND
    sys_prompt = "根据以上对话中阅读的文言文译文和分句后的选项，选择一个符合题意的选项。可以利用当前对话中的相关证据进行判断，但请注意证据不一定都有用，必须结合你自己的知识和文章进行判断。不需要做任何解释，只返回选项对应的字母A, B, C或D。"
    
    prompt = f"{SS}题目: {a}\n{segmented_options}\n\n{SS}证据:{example}\n{c}\n\n"
    formatted_sample = sys_prompt + "\n\n" + prompt
    
    ans_mes = {'role': 'user', 'content': formatted_sample}
    mes.append(ans_mes)
    res, mes = chat(mes)
    # GET THE ANSWER
    if res.status_code == HTTPStatus.OK:
        response_text = res.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            res.request_id, res.status_code, res.code, res.message
        ))
        
    results.append(response_text)