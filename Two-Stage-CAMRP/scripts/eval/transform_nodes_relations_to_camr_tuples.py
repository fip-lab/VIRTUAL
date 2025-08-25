class camr_tuple:
    def __init__(self):
        self.sid = "-"
        self.nid1 = "-"
        self.concept1 = "-"
        self.coref1 = "-"
        self.rel = "-"
        self.rid = "-"
        self.ralign = "-"
        self.nid2 = "-"
        self.concept2 = "-"
        self.coref2 = "-"
    def __str__(self):
        return '\t'.join([self.sid, self.nid1, self.concept1, self.coref1, self.rel, self.rid, self.ralign, self.nid2, self.concept2, self.coref2])


def transform_to_camr_tuples(relation_path,node_path,id_path=None):
    with open(relation_path,"r") as f1:
        relations = f1.readlines()
    with open(node_path,"r") as f2:
        nodes = f2.readlines()

    if id_path:
        with open(id_path,"r") as f2:
            id_from_files = f2.readlines()
    

    nodes_list = [eval(i.replace("(\'b","(\'x")) for i in nodes]
    
    relation_list = [ eval(i.strip().split("\t")[1])  for i in relations]
    if id_path == None:
        id_list = [ i.split("\t")[0] for i in relations]
    else:
        id_list = [ i.split("\t")[0] for i in id_from_files]

    output = open(relation_path+".camr_tuple","w")
    output.write("句子编号	节点编号1	概念1	同指节点1	关系	关系编号	关系对齐词	节点编号2	概念2	同指节点2\n")
    output.write("sid	nid1	concept1	coref1	rel	rid	ralign	nid2	concept2	coref2\n")
    

    for nods,rels,id_c in zip(nodes_list,relation_list,id_list):
        # first all nodes
        # then all relations
        nodes_tuples=[]
        edges_tuples=[]

        align_info = {i[0]:i[1] for i in nods}
        align_appear_flag = {i[0]:0 for i in nods}
        
        for i in rels:
            c_tuple = camr_tuple()
            c_tuple.sid = id_c
            if len(i)==3:
                c_tuple.nid1 = i[0]
                c_tuple.concept1 = align_info[i[0]]
                c_tuple.rel = i[1]
                c_tuple.nid2 = i[2]
                c_tuple.concept2 = align_info[i[2]]

                align_appear_flag[i[0]] = 1
                align_appear_flag[i[2]] = 1

            elif len(i)==5:
                # ['x22', ':arg0', 'x23', '有', 'x92']
                try:
                    c_tuple.nid1 = i[0]
                    c_tuple.concept1 = align_info[i[0]]
                    c_tuple.rel = i[1]
                    c_tuple.nid2 = i[4]
                    c_tuple.concept2 = align_info[i[4]]
                    c_tuple.ralign = i[3]
                    c_tuple.rid= i[2]
                except Exception as e:
                    print(e)
                    continue

                align_appear_flag[i[0]] = 1
                align_appear_flag[i[4]] = 1
            else:
                print(i)
                raise Exception("wrong relation length")
            
            edges_tuples.append(c_tuple)

        for i in nods:
            if i[1]=="root":
                continue
            
            if align_appear_flag[i[0]] == 1:
                continue
            
            c_tuple = camr_tuple()
            c_tuple.sid = id_c
            c_tuple.nid1 = i[0]
            c_tuple.nid2 = i[0]
            c_tuple.concept1 = i[1]
            c_tuple.concept2 = i[1]
            nodes_tuples.append(c_tuple)
        
        #start outputing amr graphs

        output.write("\n")
        for i in edges_tuples:
            output.write(str(i)+"\n")
        for i in nodes_tuples:
            output.write(str(i)+"\n")
        
    output.write("\n")
        



import sys
# test_id="/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/test/test_B/test_B.txt"
test_id="../../test_A/virtual_test_with_id.txt"
transform_to_camr_tuples(sys.argv[1],sys.argv[2],test_id)
        




        
