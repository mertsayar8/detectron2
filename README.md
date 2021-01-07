

---

This project focuses on cross domain Document Object Detection (DOD). DOD is the task of decomposing a document page image into structural and logical units such as texts, titles, lists, figures. In this paper, recent research on this task is discussed, two recent datasets used in research namely Publaynet and  PRImA Layout Analysis are summarized and a method based on Mask RCNN and feature pyramid networks is given using recently released large-scaled dataset Publaynet training data for cross domain DOD task on PRImA Layout Analysis Dataset. Results on PRImA and validation set of Publaynet are compared.

This project is forked from https://github.com/hpanwar08/detectron2 and uses the pre-trained network on [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset.

PRImA dataset can be reached from https://www.primaresearch.org/datasets/Layout_Analysis.

Conversion from PRImA PAGEXML to COCOJSON format can be done with running convert_prima_to_coco.py script. Input of the script should be --prima_datapath "path_to_your_folder". Your PRImA folder should have XML and Images folders as subfolders. After running the script, final destinations of the folders should be like:
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


