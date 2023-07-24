# TensorRT_ASP


### 0. Introduction
- Goal : model compression using by Structured Sparsity
- Base Model : Resnet18
- Dataset : Imagenet100
- Pruning Process :
    1. Train base model with Imagenet100 dataset
    2. Prune the model in a 2:4 sparse pattern for the FC and convolution layers.
    3. Retrain the pruned model
    4. Convert to TensorRT PTQ int8 Model
---

### 1. Development Environment
- Device
  - Windows 10 laptop
  - CPU i7-11375H
  - GPU RTX-3060
- Dependency
  - cuda 12.1
  - cudnn 8.9.2
  - tensorrt 8.6.1
  - pytorch 2.1.0+cu121

---

### 2. Code Scheme
```
    Quantization_EX/
    ├── calibrator.py       # calibration class for TensorRT PTQ
    ├── common.py           # utils for TensorRT
    ├── onnx_export.py      # onnx export ASP model
    ├── train.py            # base model train with ASP 
    ├── trt_infer_2.py      # TensorRT model build using Polygraphy
    ├── trt_infer_acc.py    # TensorRT model accuracy check
    ├── trt_infer.py        # TensorRT model infer
    ├── utils.py            # utils
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Calculation 10000 iteration with one input data [1, 3, 224, 224]

<table border="0"  width="100%">
    <tbody align="center">
        <tr>
            <td></td>
            <td><strong>TensorRT PTQ</strong></td>
            <td><strong>TensorRT PTQ with ASP</strong></td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>Int8</td>
            <td>Int8</td>
        </tr>
        <!-- <tr>
            <td>Acc Top-1 [%] </td>
            <td>  83.12  </td>
            <td>  83.18  </td>
        </tr> -->
        <tr>
            <td>Avg Latency [ms]</td>
            <td>  0.418 ms </td>
            <td>  0.388 ms </td>
        </tr>
        <tr>
            <td>Avg FPS [frame/sec]</td>
            <td> 2388.33 fps </td>
            <td> 2572.17 fps </td>
        </tr>
        <tr>
            <td>Gpu Memory [MB]</td>
            <td>  123 MB </td>
            <td>  119 MB </td>
        </tr>
    </tbody>
</table>

### 4. Guide
- train -> onnx_export -> trt_infer -> trt_infer_acc

### 5. Reference
* ASP (Automatic SParsity) : https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity
* Polygraphy : https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>
