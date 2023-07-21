import tensorrt as trt
import pycuda.driver as cuda

#from infer_engine import infer
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, SaveEngine
from polygraphy.logger import G_LOGGER
from utils import *
import common
from PIL import Image
import json
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
# TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

set_random_seeds()
device = device_check()

# 0. dataset
batch_size = 1
workers = 1

transform_ = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_dataset = datasets.ImageFolder("/mnt/h/dataset/imagenet100/val", transform=transform_)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=False,
    sampler=None,
    drop_last=True
)


# Data loader argument to `Calibrator` 
def calib_data(val_batches, input_name):
    for iteration, (images, labels) in enumerate(val_batches):
        yield {input_name: images.numpy()}
 
# Set path to ONNX model
model_name = "resnet18_1_pruned"
onnx_path = f"onnx_model/{model_name}.onnx"
engine_path = f"trt_model/{model_name}_2.trt"

if 0 :
    # Set calibrator
    calibration_cache_path = onnx_path.replace(".onnx", "_calibration.cache")
    calibrator = Calibrator(
        data_loader=calib_data(val_loader, 'input'), 
        cache=calibration_cache_path
    )
    
    # Build engine from ONNX model by enabling INT8 and sparsity weights, and providing the calibrator
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(onnx_path),
        config=CreateConfig(
            fp16=True,
            int8=True,
            calibrator=calibrator,
            builder_optimization_level=5,
            sparse_weights=True
        )
    )
 
    # Trigger engine saving
    build_engine = SaveEngine(build_engine, path=engine_path)

    # Calibrate engine (activated by the runner)
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        print("Calibrated engine!")

dur_time = 0
iteration = 10000

# 1. input
test_path = "/mnt/h/dataset/imagenet100/val/n02077923/ILSVRC2012_val_00023081.JPEG"
img = Image.open(test_path)
img = transform_(img).unsqueeze(dim=0)
input_host = np.array(img, dtype=np.float32, order="C")

classes = val_dataset.classes
class_to_idx = val_dataset.class_to_idx
class_count = len(classes)

json_file = open("/mnt/h/dataset/imagenet100/Labels.json")
class_name = json.load(json_file)

# Output shapes expected by the post-processor
output_shapes = [(1, class_count)]

# Infer PTQ engine and evaluate its accuracy
# If a serialized engine exists, use it instead of building an engine.
print(f"Reading engine from file {engine_path}")
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = input_host

        # warm-up
        for _ in range(100):
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        torch.cuda.synchronize()

        for i in range(iteration):
            begin = time.time()
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            torch.cuda.synchronize()
            dur = time.time() - begin
            dur_time += dur

        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        t_outputs = [output.reshape(shape) for output, shape in zip(t_outputs, output_shapes)]

        # 3. results
        print(engine_path)
        print(f"Using precision int8 mode.")
        print(f"{iteration}th iteration time : {dur_time} [sec]")
        print(f"Average fps : {1/(dur_time/iteration)} [fps]")
        print(f"Average inference time : {(dur_time/iteration) * 1000} [msec]")
        max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
        max_value = max_tensor[0].cpu().data.numpy()[0]
        max_index = max_tensor[1].cpu().data.numpy()[0]
        print(f"max index : {max_index}, value : {max_value}, class name : {classes[max_index]} {class_name.get(classes[max_index])}")



