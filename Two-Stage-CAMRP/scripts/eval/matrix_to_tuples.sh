RELATION_NODE=$1
RELATION_MATRIX=$2
NODE_NO_R=$3

echo $RELATION_MATRIX.literal

python relation_number_to_literal.py $RELATION_NODE $RELATION_MATRIX $RELATION_MATRIX.literal

python sync_relation_with_nodes.py $NODE_NO_R $RELATION_MATRIX.literal $RELATION_MATRIX.literal.sync_with_no_r

python transform_nodes_relations_to_camr_tuples.py $RELATION_MATRIX.literal.sync_with_no_r $NODE_NO_R




