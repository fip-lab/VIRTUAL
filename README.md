# A Comprehensive Literary Chinese Reading Comprehension Dataset with an Evidence Curation Based Solution

```
conda create -n acrc python=3.9
pip install -r crisis_requirement.txt
```

To directly reproduce the results, you can skip to the inference section.

### Preprocessing

#### AMR

https://github.com/pkunlp-icler/Two-Stage-CAMRP

```
conda create -n camrp python=3.8
conda activate camrp
pip install -r requirement.txt
```

- Replace `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/models/bert/modeling_bert.py` with `./src/modeling_bert.py`
- Replace `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/modeling_outputs.py` with `./src/modeling_outputs.py`
- Replace `/miniconda3/envs/camrp/lib/python3.7/site-packages/transformers/trainer.py` with`./src/trainer.py`

Download five models from Google Drive and place them in `./Two-Stage-CAMRP/models/trained_models`.

```
/Two-Stage-CAMRP/models
└─trained_models
    ├─non_aligned_tagging
    │  └─checkpoint-1400
    ├─normalization_tagging
    │  └─checkpoint-650
    ├─relation_align_cls
    │  └─checkpoint-33000
    ├─relation_cls
    │  └─checkpoint-32400
    └─surface_tagging
        └─checkpoint-125200
```

```
export CUDA_VISIBLE_DEVICES=0

cd ./Two-Stage-CAMRP/scripts/eval

python inference_surface_tagging.py ../../models/trained_models/surface_tagging/checkpoint-125200 ../../test_A/virtual_test_with_id.txt ../../result/virtual_test

python inference_normalization_tagging.py ../../models/trained_models/normalization_tagging/checkpoint-650 ../../test_A/virtual_test_with_id.txt ../../result/virtual_test

python inference_non_aligned_tagging.py ../../models/trained_models/non_aligned_tagging/checkpoint-1400 ../../test_A/virtual_test_with_id.txt ../../result/virtual_test

bash inference.sh ../../result/virtual_test.surface ../../result/virtual_test.norm_tag ../../result/virtual_test.non_aligned ../../test_A/virtual_test.txt ../../result/virtual_test ../../models/trained_models/relation_cls/checkpoint-32400 ../../models/trained_models/relation_align_cls/checkpoint-33000
```

Convert the AMR triples into natural language.

```
cd ./Two-Stage-CAMRP/test_A
python tuple_to_clause.py
```

#### Evidence extraction

```
conda activate acrc
cd ./VIRTUAL
python evidence_extraction.py
```

#### ZDIC

```
pip install jiayan 
pip install https://github.com/kpu/kenlm/archive/master.zip

cd ./acrc_word_segmentation

python acrc_train_word_segmentation.py
python zdic.py
```

### Inference

```
cd ./VIRTUAL
python one_shot.py --top_m=1 --top_n=3
```

