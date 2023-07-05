### **Environment：**

```
conda create -name pysgg python=3.8
conda activate pysgg
pip install torch==1.9.11+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

install mmcv and mmdet：

```
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install openmim
mim install mmdet==2.25.1
pip install git+https://github.com/cocodataset/panopticapi.git

conda install -c conda-forge pycocotools
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

pip install -v -e .
```

### Data：

```
Openpsg
----...
----data
	----coco
		----panoptic_train2017
		----panoptic_val2017
		----train2017
		----val2017
	----psg
		----psg_train_val.json
		----psg_val_test.json
		----psg_val.json
		----pag_test.json
		...
----...
```
### Pre-Train Model：

you should first download some off-the-shelf panoptic head in /work_dirs/checkpoints.

we have put the Mask2Former(ResNet50) in the project.

### Main method mentioned in our paper:
proposal matching: 
```python
model.train_cfg.use_proposal_matching=True
```
propsal feature:
```python
model.relation_head.head_config.feature_extract_method='query' # for training original framework, just set 'roi'
```

relation-constrained inferencec:
```python
model.test_cfg.use_peseudo_inference=True
```

### Training：
```python
bash scripts/mask2former/train.sh
```
(single GPU can also be trained in a similar performance presented in our paper)

### Inference：
setup the config file and checkpoint in tools/test.py
```python
python tools/test.py
```
