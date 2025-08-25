import enum

from pyparsing import nums

from decimal import *
import cn2an

from concept_norm_v6 import normalize_concept_sentence,concepts,align_sent_train,get_similar_concept

import re

def normalize_concept_v2(i, concept_list):
    """
    新增一个参数concepts: camr_pred_dict.xls中第一列所有的词 通过get_camr_dict函数获得
    """
    i = i.replace("两", "二")
    if i in align_ws_train.keys():
        train_sense = sorted(align_ws_train[i].items(),key=lambda x:-x[1])
    else:
        if(i != cn2an.transform(i)):
            train_sense_str = cn2an.transform(i)
        elif(i in CN_UNIT.keys()):
            train_sense_str = str(CN_UNIT[i])
        elif(i[-1] in CN_UNIT.keys()):
            train_sense_str = str(round(float(cn2an.transform(i[:-1])) * CN_UNIT[i[-1]]))
        else:
            train_sense_str = i
        
        if(re.findall('\d+',train_sense_str) != []):
            train_sense_str = filter_num(train_sense_str)
            train_sense = [(train_sense_str, 1)]
        else:
            if i in concept_list:
                train_sense = [(i + '-01', 1)]
            else:
                train_sense_str, similarity = get_similar_concept(i, concept_list)
                if(similarity < 0.2):
                    train_sense = [(i, similarity)]
                else:
                    train_sense = [(train_sense_str + '-01', similarity)]
    return train_sense


CN_UNIT = {
        '十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '万': 10000, '萬': 10000,
        '亿': 100000000, '億': 100000000, '兆': 1000000000000
    }

def filter_num(string):
    restr = ''
    for i in string:
        if i >= '0' and i <= '9' or i == '.':
            restr += i
    return restr

def compute_word_sense(path):
    with open(path,"r") as f:
        lines = [eval(i.strip()) for i in f.readlines()]
    align_ws = {}
    for i in lines:
        for j in i:
            if j[0] not in align_ws.keys():
                align_ws[j[0]] = {j[1]:1}
            else:
                if j[1] not in align_ws[j[0]].keys():
                    align_ws[j[0]][j[1]] = 1
                else:
                    align_ws[j[0]][j[1]] += 1
    return align_ws

def process_split_concept(word,windex):
    print(word)
    result = []

    if "两"==word:
        result.append(["x"+str(windex),"2"]) 

    if "多" in word:
        index = word.index("多")+1
        result.append(["x"+str(windex)+"_"+str(index),"多"])
        if len(word) > 1:
            else_number = normalize_concept(word.replace("多",""))[0][0]
            number_string= "_".join([ str(i+1) for i in range(len(word)) if i+1 != index])
            result.append(["x"+str(windex)+"_"+number_string,else_number])
    if "几" in word:
        index = word.index("几")+1
        else_number = normalize_concept(word.replace("几",""))[0][0]
        result.append(["x"+str(windex)+"_"+str(index),"几"])
        number_string= "_".join([ str(i+1) for i in range(len(word)) if i+1 != index])
        result.append(["x"+str(windex)+"_"+number_string,else_number])
    if "余" in word:
        index = word.index("余")+1
        else_number = normalize_concept(word.replace("余",""))[0][0]
        result.append(["x"+str(windex)+"_"+str(index),"余"])
        number_string= "_".join([ str(i+1) for i in range(len(word)) if i+1 != index])
        result.append(["x"+str(windex)+"_"+number_string,else_number])
    if "数" in word:
        index = word.index("数")+1
        else_number = normalize_concept(word.replace("数",""))[0][0]
        result.append(["x"+str(windex)+"_"+str(index),"数"])
        number_string= "_".join([ str(i+1) for i in range(len(word)) if i+1 != index])
        result.append(["x"+str(windex)+"_"+number_string,else_number])
    
    if "没有" in word:
        result.append(["x"+str(windex)+"_1","-"])
        result.append(["x"+str(windex)+"_2","有-03"])
    if "最" in word:
        result.append(["x"+str(windex)+"_1","最"])
        if len(word)>1:
            result.append(["x"+str(windex)+"_2",word[1]+"-01"])
    if "较" in word:
        result.append(["x"+str(windex)+"_1","较"])

        if len(word)>1:
            result.append(["x"+str(windex)+"_2",word[1]+"-01"])
    
    return result


def transform_nodes_with_mp_v6(concept_tag_path,transform_tag_path,input_words_path,output_path):
    # use 8 classes taging, v6 wsd
    # 相比v5 使用新的wsd
    concepts_output = []
    concepts_output_no_r = []
    total_split=0
    processed_split=0

    with open(input_words_path,"r") as f:
        words = [ i.strip().split(" ") for i in f.readlines() ]
    
    with open(concept_tag_path,"r") as f:
        concept_tags = [ i.strip().split(" ") for i in f.readlines() ]

    with open(transform_tag_path,"r") as f:
        transform_tags = [ i.strip().split(" ") for i in f.readlines() ]
    

    for word,concept,trans in zip(words,concept_tags,transform_tags):
        output_concept_alignment=[['x0','root']]
        output_concept_alignment_with_ralign=[['x0','root']]
        assert len(word)==len(concept)==len(trans)

        ori_word = [i for i in word]

        # first do wsd
        word_number={}

        for i,tag in enumerate(trans):
            if tag == '1':
                # need transform
                if word[i] in word_number.keys():
                    word_number[word[i]] += 1
                else:
                    word_number[word[i]] =1
                word[i] = normalize_concept_sentence(ori_word[i],"".join(ori_word),concepts, align_sent_train, word_number[ori_word[i]])[0][0]



       
        # extract concepts
        # id:  label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
        # B-SPLIT 不处理，直接提交

        current_ralign_concept=[]
        current_concept=[]
        current_cont_concept=[]
        current_unc_concept=[]
        split_concept=[]
        for i,tag in enumerate(concept):
            if tag=='0':
                #提交所有
                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]

            if tag=='1':
                # 直接提交
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='2':
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=["x"+str(i+1),word[i]]
                else:
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='3':
                if len(current_cont_concept)!=0:
                    current_cont_concept[0]+="_x"+str(i+1)
                    current_cont_concept[1]+=word[i]
                else:
                    "wrong tag, regard as 2"
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='4':
                if len(current_unc_concept)!=0:
                    output_concept_alignment.append(current_unc_concept)
                    output_concept_alignment_with_ralign.append(current_unc_concept)
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept=["x"+str(i+1),word[i]]

                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='5':
                if len(current_unc_concept)==0:
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept[0]+="_x"+str(i+1)
                    current_unc_concept[1]+=word[i]
            
            if tag=='6':
                total_split+=1
                result = process_split_concept(word[i],i+1)
                if result!=[]:
                    processed_split+=1
                    output_concept_alignment = output_concept_alignment + result
                else:
                    print(word[i],"split tag not process")
                    output_concept_alignment.append(["x"+str(i+1),word[i]])
                    output_concept_alignment_with_ralign.append(["x"+str(i+1),word[i]])
                    print(output_concept_alignment)
            
            if tag=='7':
                # 直接提交 给虚词concept
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
        
        if len(current_concept)!=0:
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
        if len(current_unc_concept)!=0:
                output_concept_alignment.append(current_unc_concept)
                output_concept_alignment_with_ralign.append(current_unc_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        
        concepts_output_no_r.append(output_concept_alignment)
        concepts_output.append(output_concept_alignment_with_ralign)

    print("total split and processed split:",total_split,processed_split)
    with open(output_path+".mpv5.no_r","w") as f:
        for i in concepts_output_no_r:
            f.write(str(i)+"\n")
    
    with open(output_path+".mpv5.with_r","w") as f:
        for i in concepts_output:
            f.write(str(i)+"\n")


def transform_nodes_with_mp_v7(concept_tag_path,transform_tag_path,input_words_path,output_path):
    # use 8 classes taging, v2 wsd
    # 相比v4 添加了对虚词独立标记，生成两个文件，一个含有虚词，一个不含有虚词
    concepts_output = []
    concepts_output_no_r = []
    total_split=0
    processed_split=0

    with open(input_words_path,"r") as f:
        words = [ i.strip().split(" ") for i in f.readlines() ]
    
    with open(concept_tag_path,"r") as f:
        concept_tags = [ i.strip().split(" ") for i in f.readlines() ]

    with open(transform_tag_path,"r") as f:
        transform_tags = [ i.strip().split(" ") for i in f.readlines() ]
    

    for word,concept,trans in zip(words,concept_tags,transform_tags):
        output_concept_alignment=[['x0','root']]
        output_concept_alignment_with_ralign=[['x0','root']]
        assert len(word)==len(concept)==len(trans)

        # first do wsd
        for i,tag in enumerate(trans):
            if tag == '1':
                # need transform
                word[i] = normalize_concept_v2(word[i],concepts)[0][0]
        # extract concepts
        # id:  label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
        # B-SPLIT 不处理，直接提交

        current_ralign_concept=[]
        current_concept=[]
        current_cont_concept=[]
        current_unc_concept=[]
        split_concept=[]
        for i,tag in enumerate(concept):
            if tag=='0':
                #提交所有
                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]

            if tag=='1':
                # 直接提交
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='2':
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=["x"+str(i+1),word[i]]
                else:
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='3':
                if len(current_cont_concept)!=0:
                    current_cont_concept[0]+="_x"+str(i+1)
                    current_cont_concept[1]+=word[i]
                else:
                    "wrong tag, regard as 2"
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]

            
            if tag=='4':
                if len(current_unc_concept)!=0:
                    output_concept_alignment.append(current_unc_concept)
                    output_concept_alignment_with_ralign.append(current_unc_concept)
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept=["x"+str(i+1),word[i]]

                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='5':
                if len(current_unc_concept)==0:
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept[0]+="_x"+str(i+1)
                    current_unc_concept[1]+=word[i]
            
            if tag=='6':
                total_split+=1
                result = process_split_concept(word[i],i+1)
                if result!=[]:
                    processed_split+=1
                    output_concept_alignment = output_concept_alignment + result
                else:
                    print(word[i],"split tag not process")
                    output_concept_alignment.append(["x"+str(i+1),word[i]])
                    output_concept_alignment_with_ralign.append(["x"+str(i+1),word[i]])
                    print(output_concept_alignment)
            
            if tag=='7':
                # 直接提交 给虚词concept
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
        

        


        

        if len(current_concept)!=0:
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
        if len(current_unc_concept)!=0:
                output_concept_alignment.append(current_unc_concept)
                output_concept_alignment_with_ralign.append(current_unc_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        
        concepts_output_no_r.append(output_concept_alignment)
        concepts_output.append(output_concept_alignment_with_ralign)

    print("total split and processed split:",total_split,processed_split)
    with open(output_path+".no_r","w") as f:
        for i in concepts_output_no_r:
            f.write(str(i)+"\n")
    
    with open(output_path+".with_r","w") as f:
        for i in concepts_output:
            f.write(str(i)+"\n")


def transform_nodes_with_mp_v5(concept_tag_path,transform_tag_path,input_words_path,output_path):
    # use 8 classes taging, v2 wsd
    # 相比v4 添加了对虚词独立标记，生成两个文件，一个含有虚词，一个不含有虚词
    concepts_output = []
    concepts_output_no_r = []
    total_split=0
    processed_split=0

    with open(input_words_path,"r") as f:
        words = [ i.strip().split(" ") for i in f.readlines() ]
    
    with open(concept_tag_path,"r") as f:
        concept_tags = [ i.strip().split(" ") for i in f.readlines() ]

    with open(transform_tag_path,"r") as f:
        transform_tags = [ i.strip().split(" ") for i in f.readlines() ]
    

    for word,concept,trans in zip(words,concept_tags,transform_tags):
        output_concept_alignment=[['x0','root']]
        output_concept_alignment_with_ralign=[['x0','root']]
        assert len(word)==len(concept)==len(trans)

        # first do wsd
        for i,tag in enumerate(trans):
            if tag == '1':
                # need transform
                word[i] = normalize_concept(word[i])[0][0]
        # extract concepts
        # id:  label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
        # B-SPLIT 不处理，直接提交

        current_ralign_concept=[]
        current_concept=[]
        current_cont_concept=[]
        current_unc_concept=[]
        split_concept=[]
        for i,tag in enumerate(concept):
            if tag=='0':
                #提交所有
                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]

            if tag=='1':
                # 直接提交
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='2':
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=["x"+str(i+1),word[i]]
                else:
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='3':
                if len(current_cont_concept)!=0:
                    current_cont_concept[0]+="_x"+str(i+1)
                    current_cont_concept[1]+=word[i]
                else:
                    "wrong tag, regard as 2"
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]

            
            if tag=='4':
                if len(current_unc_concept)!=0:
                    output_concept_alignment.append(current_unc_concept)
                    output_concept_alignment_with_ralign.append(current_unc_concept)
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept=["x"+str(i+1),word[i]]

                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    output_concept_alignment_with_ralign.append(current_concept)
                    current_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
            
            if tag=='5':
                if len(current_unc_concept)==0:
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept[0]+="_x"+str(i+1)
                    current_unc_concept[1]+=word[i]
            
            if tag=='6':
                total_split+=1
                result = process_split_concept(word[i],i+1)
                if result!=[] and len(result)>1:
                    processed_split+=1
                    output_concept_alignment = output_concept_alignment + result
                elif  result!=[] and len(result)==1:
                    processed_split+=1
                    output_concept_alignment += result
                    output_concept_alignment_with_ralign += result
                else:
                    print(word,word[i],"split tag not process")
                    output_concept_alignment.append(["x"+str(i+1),word[i]])
                    output_concept_alignment_with_ralign.append(["x"+str(i+1),word[i]])
            
            if tag=='7':
                # 直接提交 给虚词concept
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment_with_ralign.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    output_concept_alignment_with_ralign.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    output_concept_alignment_with_ralign.append(split_concept)
                    split_concept=[]
        

        


        

        if len(current_concept)!=0:
                output_concept_alignment.append(current_concept)
                output_concept_alignment_with_ralign.append(current_concept)
        if len(current_unc_concept)!=0:
                output_concept_alignment.append(current_unc_concept)
                output_concept_alignment_with_ralign.append(current_unc_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
                output_concept_alignment_with_ralign.append(split_concept)
        
        concepts_output_no_r.append(output_concept_alignment)
        concepts_output.append(output_concept_alignment_with_ralign)

    print("total split and processed split:",total_split,processed_split)
    with open(output_path+".no_r","w") as f:
        for i in concepts_output_no_r:
            f.write(str(i)+"\n")
    
    with open(output_path+".with_r","w") as f:
        for i in concepts_output:
            f.write(str(i)+"\n")


def transform_nodes_with_mp_v4(concept_tag_path,transform_tag_path,input_words_path,output_path):
    # use 7 classes taging, v2 wsd
    # 相比v3 添加了对split_word的处理
    concepts_output = []
    total_split=0
    processed_split=0

    with open(input_words_path,"r") as f:
        words = [ i.strip().split(" ") for i in f.readlines() ]
    
    with open(concept_tag_path,"r") as f:
        concept_tags = [ i.strip().split(" ") for i in f.readlines() ]

    with open(transform_tag_path,"r") as f:
        transform_tags = [ i.strip().split(" ") for i in f.readlines() ]
    

    for word,concept,trans in zip(words,concept_tags,transform_tags):
        output_concept_alignment=[['x0','root']]
        assert len(word)==len(concept)==len(trans)

        # first do wsd
        for i,tag in enumerate(trans):
            if tag == '1':
                # need transform
                word[i] = normalize_concept(word[i])[0][0]
        # extract concepts
        # id:  label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
        # B-SPLIT 不处理，直接提交

        current_concept=[]
        current_cont_concept=[]
        current_unc_concept=[]
        split_concept=[]
        for i,tag in enumerate(concept):
            if tag=='0':
                #提交所有
                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    current_concept=[]
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=[]
                if len(split_concept)!=0:
                        output_concept_alignment.append(split_concept)
                        split_concept=[]

            if tag=='1':
                # 直接提交
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='2':
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=["x"+str(i+1),word[i]]
                else:
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='3':
                if len(current_cont_concept)!=0:
                    current_cont_concept[0]+="_x"+str(i+1)
                    current_cont_concept[1]+=word[i]
                else:
                    "wrong tag, regard as 2"
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]

            
            if tag=='4':
                if len(current_unc_concept)!=0:
                    output_concept_alignment.append(current_unc_concept)
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept=["x"+str(i+1),word[i]]

                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    current_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='5':
                if len(current_unc_concept)==0:
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept[0]+="_x"+str(i+1)
                    current_unc_concept[1]+=word[i]
            
            if tag=='6':
                total_split+=1
                result = process_split_concept(word[i],i+1)
                if result!=[]:
                    processed_split+=1
                    output_concept_alignment = output_concept_alignment + result
                else:
                    print(word[i],"split tag not process")
                    output_concept_alignment.append(["x"+str(i+1),word[i]])
                    print(output_concept_alignment)
        

        


        

        if len(current_concept)!=0:
                output_concept_alignment.append(current_concept)
        if len(current_unc_concept)!=0:
                output_concept_alignment.append(current_unc_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
        
        concepts_output.append(output_concept_alignment)

    print("total split and processed split:",total_split,processed_split)
    with open(output_path,"w") as f:
        for i in concepts_output:
            f.write(str(i)+"\n")

def transform_nodes_with_mp_v3(concept_tag_path,transform_tag_path,input_words_path,output_path):
    # use 7 classes taging, v2 wsd
    concepts_output = []

    with open(input_words_path,"r") as f:
        words = [ i.strip().split(" ") for i in f.readlines() ]
    
    with open(concept_tag_path,"r") as f:
        concept_tags = [ i.strip().split(" ") for i in f.readlines() ]

    with open(transform_tag_path,"r") as f:
        transform_tags = [ i.strip().split(" ") for i in f.readlines() ]
    

    for word,concept,trans in zip(words,concept_tags,transform_tags):
        output_concept_alignment=[['x0','root']]
        assert len(word)==len(concept)==len(trans)

        # first do wsd
        for i,tag in enumerate(trans):
            if tag == '1':
                # need transform
                word[i] = normalize_concept(word[i])[0][0]
        # extract concepts
        # id:  label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
        # B-SPLIT 不处理，直接提交

        current_concept=[]
        current_cont_concept=[]
        current_unc_concept=[]
        split_concept=[]
        for i,tag in enumerate(concept):
            if tag=='0':
                #提交所有
                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    current_concept=[]
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=[]
                if len(split_concept)!=0:
                        output_concept_alignment.append(split_concept)
                        split_concept=[]

            if tag=='1':
                # 直接提交
                current_concept=["x"+str(i+1),word[i]]
                output_concept_alignment.append(current_concept)
                current_concept=[]

                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='2':
                if len(current_cont_concept)!=0:
                    output_concept_alignment.append(current_cont_concept)
                    current_cont_concept=["x"+str(i+1),word[i]]
                else:
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='3':
                if len(current_cont_concept)!=0:
                    current_cont_concept[0]+="_x"+str(i+1)
                    current_cont_concept[1]+=word[i]
                else:
                    "wrong tag, regard as 2"
                    current_cont_concept=["x"+str(i+1),word[i]]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]

            
            if tag=='4':
                if len(current_unc_concept)!=0:
                    output_concept_alignment.append(current_unc_concept)
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept=["x"+str(i+1),word[i]]

                if len(current_concept)!=0:
                    output_concept_alignment.append(current_concept)
                    current_concept=[]
                
                if len(split_concept)!=0:
                    output_concept_alignment.append(split_concept)
                    split_concept=[]
            
            if tag=='5':
                if len(current_unc_concept)==0:
                    current_unc_concept=["x"+str(i+1),word[i]]
                else:
                    current_unc_concept[0]+="_x"+str(i+1)
                    current_unc_concept[1]+=word[i]
            
            if tag=='6':
                #print(print(word[i]))
                pass
        

        if len(current_concept)!=0:
                output_concept_alignment.append(current_concept)
        if len(current_unc_concept)!=0:
                output_concept_alignment.append(current_unc_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
        if len(split_concept)!=0:
                output_concept_alignment.append(split_concept)
        
        concepts_output.append(output_concept_alignment)
    
    with open(output_path,"w") as f:
        for i in concepts_output:
            f.write(str(i)+"\n")



#compute baseline acc for wsd and none wsd : concept normalization
total_num=0
match_num=0

none_wsd=0
match_none_wsd=0

align_ws_train = compute_word_sense("./wsd_dataset/train_clean.p_align")
#align_ws_dev = compute_word_sense("./v1_4classes/dev.p_align")

def normalize_concept(i):
    if not i:  # Check if the string is empty
        return [("", "")]
    if(i.isdigit()):
        return [(i,"")]
    elif(i in CN_UNIT.keys()):
        return [(str(CN_UNIT[i]),"")]
    elif(len(i) > 0 and i[-1] in CN_UNIT.keys()):
        try:
            return [(str(round(float(cn2an.transform(i[:-1])) * CN_UNIT[i[-1]])),"")]
        except:
            return [(i,"")]
    else:
        return [(i,"-01")]

import sys
concept_tag_path=sys.argv[1]
transform_tag_path=sys.argv[2]
input_words_path=sys.argv[3]
output_path=sys.argv[4]
transform_nodes_with_mp_v5(concept_tag_path,transform_tag_path,input_words_path,output_path)