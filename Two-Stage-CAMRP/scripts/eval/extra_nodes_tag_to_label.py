def add_extra_nodes(tag_path,nodes_path,out_path):
    tag_2_label={}
    with open("../../preprocessed/non_aligned_concept_tagging/train.tag.extra_nodes_dict","r") as f:
        for i,j in enumerate(f.readlines()):
            tag_2_label[str(i+1)] = j.strip()
    raw_nodes=[]
    new_nodes=[]
    tags=[]
    with open(nodes_path,"r") as f:
        raw_nodes = [eval(i) for i in f.readlines()]
    with open(tag_path,"r") as f:
        tags = [i.strip().split() for i in f.readlines()]
    

    for nod,tag in zip(raw_nodes,tags):
        index=1000
        for j in tag:
            if j!="0":
                nod.append(["x"+str(index),tag_2_label[j]])
                index += 1
        new_nodes.append(nod)
    
    with open(out_path,"w") as f:
        for i in new_nodes:
            f.write(str(i)+"\n")

import sys

EXTRA_LABEL_TAG=sys.argv[1]
ALIGEN_LABEL_FILE=sys.argv[2]
OUT_PUT=sys.argv[3]

add_extra_nodes(EXTRA_LABEL_TAG,ALIGEN_LABEL_FILE,OUT_PUT)

