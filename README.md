

---

This project focuses on cross domain Document Object Detection (DOD). DOD is the task of decomposing a document page image into structural and logical units such as texts, titles, lists, figures. In this paper, recent research on this task is discussed, two recent datasets used in research namely Publaynet and  PRImA Layout Analysis are summarized and a method based on Mask RCNN and feature pyramid networks is given using recently released large-scaled dataset Publaynet training data for cross domain DOD task on PRImA Layout Analysis Dataset. Results on PRImA and validation set of Publaynet are compared.

This project is forked from https://github.com/hpanwar08/detectron2 and uses the pre-trained network on [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset.

PRImA dataset can be reached from https://www.primaresearch.org/datasets/Layout_Analysis.

Conversion from PRImA PAGEXML to COCOJSON format can be done with running convert_prima_to_coco.py script. Input of the script should be --prima_datapath "path_to_your_folder". Your PRImA folder should have XML and Images folders as subfolders. After running the script, final destination should be like:
  ```bash
        data/
        └── prima/
            ├── Images/
            ├── XML/
            ├── License.txt
            └── annotations*.json
 ```
This conversion script is updated from https://github.com/Layout-Parser/layout-model-training.

cs555_project.py script or cs555_projectFinal.ipynb notebook should be run in order to make object predictions and dataset evaluations. This project is run on Google Colab.


## Steps to test pretrained models locally or jump to next section for docker deployment
* Install the latest `Detectron2` from https://github.com/facebookresearch/detectron2
* Copy config files (`DLA_*`) from this repo to the installed Detectron2
* Download the relevant model from the `Benchmarking` section. If you have downloaded model using `wget` then refer https://github.com/hpanwar08/detectron2/issues/22
* Add the below code in demo/demo.py in the `main`to get confidence along with label names
```
from detectron2.data import MetadataCatalog
MetadataCatalog.get("dla_val").thing_classes = ['text', 'title', 'list', 'table', 'figure']
```
* Then run below command for prediction on single image (change the config file relevant to the model)
```
python demo/demo.py --config-file configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml --input "<path to image.jpg>" --output <path to save the predicted image> --confidence-threshold 0.5 --opts MODEL.WEIGHTS <path to model_final_trimmed.pth> MODEL.DEVICE cpu
```


## Sample results from detectron2

| <img src="assets/images/resnext101_32x8d/PMC1247189_00000.jpg" width=400> | <img src="assets/images/resnext101_32x8d/PMC1247608_00001.jpg" width=400> |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="assets/images/resnext101_32x8d/PMC1281292_00001.jpg" width=400> | <img src="assets/images/resnext101_32x8d/PMC1343590_00003.jpg" width=400> |
| <img src="assets/images/resnext101_32x8d/PMC2778503_00000.jpg" width=400> | <img src="assets/images/resnext101_32x8d/PMC6052416_00007.jpg" width=400> |
| <img src="assets/images/resnext101_32x8d/PMC6095069_00001.jpg" width=400> | <img src="assets/images/resnext101_32x8d/PMC6095088_00000.jpg" width=400> |
| <img src="assets/images/resnext101_32x8d/PMC6098231_00004.jpg" width=400> |                                                                           |

--- 



