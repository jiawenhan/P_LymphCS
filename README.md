# P-LymphCS

Code for the paper "Developing a Pathological Feature Extraction Network as the Backbone for the Lymphoma Classification System". This repository mainly contains the source code for lymphoma classification based on deep learning, which is used to extract the features of patch images. Subsequently, the multi-branch clustering-constrained attention multiple instance learning (CLAM) is employed to extract the features at the whole slide image (WSI) level as classifier to diagnose the lymphoma subtypes of slides. The CLAM algorithm can be found in [CLAM](https://github.com/mahmoodlab/CLAM/tree/master).

![](Data/P_LymphCS_overview.png?v=1&type=image)

## Folder Structure

It is necessary to split the WSI files into patch images with a size of 512×512 pixels in advance, and organize these files in the required format. Users are required to save all the patch images cropped from each WSI in the same folder, and there will be as many folders as the number of WSIs. The images are expected to be organized as follows:

```bash
Dataset
   ├──WSI-1
           ├──WSI-1-Patch-1.png
           ├──WSI-1-Patch-2.png
           ├──WSI-1-Patch-3.png
   ├──WSI-2
           ├──WSI-2-Patch-1.png
           ├──WSI-2-Patch-2.png
           ├──WSI-2-Patch-3.png  
```

After the dataset is well organized, it is also necessary to configure the label file, which is in the txt format. These label files contains the path of patch images and corresponding labels. The label files of the training cohort, internal testing cohort and external testing cohort are saved in the same folders named `train.txt`, `val.txt` and `test.txt` respectively. A `labels.txt` also needs to be included in this folder, in which the categories of the labels are listed. We have provided an example of a dataset and placed it in the`/Data/input_images`. We also provided the corresponding label files for reference, and these files are placed in the`/Data/Label_Files`.

## Finetune Model

The deep Learning (DL) functions of the P-LymphCS are all integrated into main.py. The weight file of backbone model is placed in the `/Data/pretrained_Weight` . Run `main.py --train` to train DL backbone model. The hyperparameters of the model can be modified in the main.py. Further hyperparameters and functions need to be modified in the `/P_LymphCS/function/run_classification.py`. The output results are defaultly stored in the `/Output/Train`. You can use following command for model training.

```shell
python main.py --train --model_name LpCTransVss --image_data ./Data/input_images --label_dir ./Data/Label_Files --model_path ./Data/pretrained_Weight
```

## Inference

Run `main.py --predict` to predict the lymphoma subtype of patch level images. The hyperparameters of the model can be modified in the main.py file. Further hyperparameters and functions need to be modified in the `/P_LymphCS/function/run_predict.py`. The output results are defaultly stored in the `/Output/Patch_Predict`. You can use following command for subtypes classification.

```Shell
python main.py --predict --model_name LpCTransVss --image_data ./Data/input_images --label_dir ./Data/Label_Files --model_path ./Data/pretrained_Weight
```
# Model Prediction CSV Format

The model produces predictions for five classes. The first row contains the column headers, and each subsequent row represents one sample.

## CSV Format

```csv
label,prob_0,prob_1,prob_2,prob_3,prob_4,predicted_label
```

| Field | Description |
|---|---|
| `label` | Ground-truth class index |
| `prob_0` | Predicted probability for class 0 |
| `prob_1` | Predicted probability for class 1 |
| `prob_2` | Predicted probability for class 2 |
| `prob_3` | Predicted probability for class 3 |
| `prob_4` | Predicted probability for class 4 |
| `predicted_label` | Final predicted class index |

## Class Mapping

| Class Index | Class Name |
|---:|---|
| `0` | `NL` |
| `1` | `TCL` |
| `2` | `HL` |
| `3` | `SBCL` |
| `4` | `LBCL` |

## Example

```csv
label,prob_0,prob_1,prob_2,prob_3,prob_4,predicted_label
0,0.70,0.10,0.10,0.05,0.05,0
1,0.05,0.80,0.05,0.05,0.05,1
2,0.10,0.10,0.65,0.10,0.05,2
3,0.05,0.10,0.10,0.70,0.05,3
4,0.05,0.05,0.10,0.10,0.70,4
```

The probability columns are generated using Softmax. The sum of `prob_0` through `prob_4` for each sample should be approximately `1`.

The value of `predicted_label` is the index of the class with the highest probability. For example, if `prob_3` is the largest value, then `predicted_label` is `3`, corresponding to the `SBCL` class.

Both `label` and `predicted_label` are stored as class indices rather than class names. Use the class mapping above to convert indices to class names.
## Extract Patch Level Features

Run `main.py --feature_extract` to extract image features. The parameters need to be modified in the `/function/Feature_Extract.py`. The output results are defaultly stored in the `/Output/Patch_Features`. You can use following command for feature extraction.

```Shell
python main.py --feature_extract --model_name LpCTransVss --image_data ./Data/input_images --label_dir ./Data/Label_Files --model_path ./Data/pretrained_Weight
```

## Visualization

Run `main.py --gradcam` to perform heatmap visualization with GradCAM method. The parameters need to be modified in the `/function/GradCam.py`. The output results are defaultly stored in the `/Output/GradCam`. You can use following command for GradCAM visualization.

```Shell
python main.py --gradcam --model_name LpCTransVss --image_data ./Data/input_images --label_dir ./Data/Label_Files --model_path ./Data/pretrained_Weight
```

## Reference and Acknowledgements

We thank the authors and developers for their contribution as below.

[CLAM](https://github.com/mahmoodlab/CLAM/tree/master)

[Twins](https://github.com/Meituan-AutoML/Twins)

[VMamba](https://github.com/MzeroMiko/VMamba)

[OneKey](https://github.com/OnekeyAI-Platform/onekey)

## License

This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
