import os

from pathlib import Path
import matplotlib.pyplot as plt
import pprint

import numpy as np

from openpsg.utils.vis_tools.datasets import coco_dir
from openpsg.utils.vis_tools.preprocess import load_json

from detectron2.data.detection_utils import read_image
from detectron2.utils.colormap import colormap
from panopticapi.utils import rgb2id

def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()

psg_dataset_file = load_json(Path("./data/psg/psg_test.json"))

psg_thing_cats = psg_dataset_file['thing_classes']
psg_stuff_cats = psg_dataset_file['stuff_classes']
psg_obj_cats = psg_thing_cats + psg_stuff_cats
psg_rel_cats = psg_dataset_file['predicate_classes']
psg_dataset = {d["image_id"]: d for d in psg_dataset_file['data']}
# psg_dataset_coco_id = {d["coco_image_id"]: d for d in psg_dataset_file['data']}

print('Number of images: {}'.format(len(psg_dataset)))
print('# Object Classes: {}'.format(len(psg_obj_cats)))
print('# Relation Classes: {}'.format(len(psg_rel_cats)))

example_img_id = '000000039785'
data = psg_dataset[example_img_id]

dir = '/home/sylvia/yjr/sgg/OpenPSG/data/coco'

img = read_image(os.path.join(dir, data["file_name"]), format="RGB")
plt.imshow(img)
plt.show()

seg_map = read_image(os.path.join(dir, data["pan_seg_file_name"]), format="RGB")
seg_map = rgb2id(seg_map)

masks = []
labels_coco = []
for i, s in enumerate(data["segments_info"]):
    label = psg_obj_cats[s["category_id"]]
    labels_coco.append(label)
    masks.append(seg_map == s["id"])

colormap_coco = get_colormap(len(data["segments_info"]))
colormap_coco = (np.array(colormap_coco) / 255).tolist()

from openpsg.utils.vis_tools.detectron_viz import Visualizer
viz = Visualizer(img)
viz.overlay_instances(
    labels=labels_coco,
    masks=masks,
    assigned_colors=colormap_coco,
)
viz_img = viz.get_output().get_image()
plt.figure(figsize=(10,10))

plt.imshow(viz_img)
plt.axis('off')
plt.savefig('temp.png')
plt.show()


