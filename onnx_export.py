#  by yhpark 2023-07-20
# tensorboard --logdir ./logs
from utils import *
import onnx
from apex.contrib.sparsity import ASP
genDir("./onnx_model")


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    data_dir = "/mnt/h/dataset/imagenet100"  # dataset path
    print(f"=> Custom {data_dir} is used!")

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=None,
    )

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_count = len(classes)

    # 1. model
    class_count = len(val_dataset.classes)
    model_name = "resnet18"
    model = models.__dict__[model_name]().to(device)
    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device)

    model_name = 'resnet18_1_pruned'
    # model_name = 'resnet18'
    check_path = f"./checkpoints/{model_name}.pth.tar"

    ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention], allow_recompute_mask=False)

    model.load_state_dict(torch.load(check_path, map_location=device))
    model.eval()

    # evaluate model status
    if False:
        test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
        print(f"acc before ptq : {test_acc1}")
        print(f"model: {model}")  # print model structure
        summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size

    # export onnx model
    export_model_path = f"./onnx_model/{model_name}.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")

    with torch.no_grad():
        torch.onnx.export(
            model,  # pytorch model
            dummy_input,  # model dummy input
            export_model_path,  # onnx model path
            opset_version=17,  # the version of the opset
            input_names=["input"],  # input name
            output_names=["output"],  # output name
            do_constant_folding=True)

        print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model check done!")


if __name__ == "__main__":
    main()
