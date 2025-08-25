def sync_relation(relation_path,node_path,out):
    with open(relation_path,"r") as f1:
        relations = f1.readlines()
    with open(node_path,"r") as f2:
        nodes = f2.readlines()
    
    new_relations = []
    ids=[]


    nodes_list = [eval(i) for i in nodes]
    relation_list = [ eval(i.split("\t")[1])  for i in relations]
    id_list = [ i.split("\t")[0] for i in relations]

    output = open(out,"w")

    for nods,rels,ids in zip(nodes_list,relation_list,id_list):
        sync_relation = []

        nod_list = [ i[0] for i in nods]

        for rel in rels:
            if len(rel) == 3:
                if rel[0] not in nod_list or rel[2] not in nod_list:
                    continue
            if len(rel) == 5:
                if rel[0] not in nod_list or rel[4] not in nod_list:
                    print(rel)
                    continue
            
            sync_relation.append(rel)
        
        output.write(ids+"\t"+str(sync_relation)+"\n")
import sys
node_path = sys.argv[1]
relation_path = sys.argv[2]
out = sys.argv[3]

sync_relation(relation_path,node_path,out)
