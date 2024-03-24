# README for Data Visualization Script

## Overview

This script is designed to visualize and compare annotations in image datasets, particularly focusing on text annotations. It can plot ground truth bounding boxes, predicted bounding boxes, and the anchors for the text regions. This visualization aids in understanding the performance of text detection models and the accuracy of annotations.

## Requirements

- Python 3.x
- Libraries: NumPy, OpenCV, Matplotlib, Pandas, NetworkX, tqdm

## Installation

Ensure that Python 3.x is installed on your system. Then, install the required Python libraries using the following command:

```sh
pip install numpy opencv-python matplotlib pandas networkx tqdm
```

## Usage

To use this script, you need to prepare your dataset and annotations according to the expected directory structure and file formats. The script expects the following inputs:

1. **Full Dataset Path (`FULL_DATA_SETS_PATH`)**: The root directory of your dataset, containing subdirectories for images, annotations, etc.
2. **Position NP (`PositionNP`)**: The subdirectory within the dataset that contains NumPy files (.npy) with anchor positions for text regions.
3. **Category CSV (`category_csv`)**: A CSV file that maps image IDs to their corresponding categories.
4. **Prediction NPY Path (`pred_npy_path`)**: The directory containing NumPy files (.npy) with predicted bounding boxes for text regions.
5. **Task List (`todo_list`)**: A text file listing the IDs of images to be processed, along with optional annotations.

### Example Usage

First, set your configuration options in the `options` dictionary. Here is an example configuration:

```python
options = {
        'denormalization':      True,                               #传入的数据是否需要反归一化, bool
        'run_model':            'Text',                             #选择运行模型。Text是画有文字内容的图片，plot是绘制ground-truth和pred的对比图，study是绘制userstudy用的图片 ['plot','study','Text']
        'dpi':                  300,                                #保存图像时用的dpi, int
        'left_or_center':       1,                                  #0是左上角，1是中心点 [0,1]
        'datasets_path':        'G:\SWUIllustration_869_v1',        #数据集的位置 string
        'postionNP_path':       'PositionNP_new2',                  #npy格式的anchor文件夹的位置 string
        'category_csv':         'category-869.csv',                 #存放类别信息的csv string
        'pred_data':            './LPGT',                           #预测的Text框，以npy的格式储存 string
        'task_list':            'new.txt',                          #要进行可视化的图片，放在Layout文件夹下 string
        'output_path':          './lpgt_output_new1',               #输出图片的存放位置 string
        'debug':                True                                #是否以debug模式运行,只画一张图，并且显示图片 bool
    }

```

After configuring the options, run the main function to start the visualization process:

```python
main(options)
```

The script will process each image listed in the `task_list` file, drawing the ground truth and predicted bounding boxes, and save the resulting images to the `output_path` directory.

## Customization

You can customize the behavior of the script by changing the values in the `options` dictionary:

- **`denormalization`**: Set to `True` if the predicted bounding box coordinates are normalized and need to be denormalized based on the image dimensions.
- **`run_model`**: Choose between `'plot'` (for comparison between ground truth and predictions), `'study'` (for predictions only), and `'Text'` (for ground truth only).
- **`dpi`**: Set the DPI (Dots Per Inch) for the output images. Higher values result in higher resolution images.
- **`left_or_center`**: Indicates whether the bounding box coordinates are based on the top-left corner (`0`) or the center (`1`).
- **`debug`**: If set to `True`, the script will run in debug mode, processing only one image and displaying it on the screen instead of saving.

## Output

The script will generate images with the visualizations specified by the `run_model` option and save them to the `output_path` directory. Each output image will correspond to an image ID listed in the `task_list` file, annotated with ground truth and/or predicted bounding boxes as per the configuration.

## Troubleshooting

- Ensure all file paths in the `options` dictionary are correct and accessible.
- Verify that the `task_list` file lists valid image IDs that exist within the dataset directory structure.
- If the script fails to read certain files or directories, check for typos in file names and paths.