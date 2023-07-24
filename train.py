#  by yhpark 2023-07-20
# tensorboard --logdir ./logs
from utils import *
from apex.contrib.sparsity import ASP
from apex.optimizers import FusedAdam
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
genDir("./checkpoints")


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    batch_size = 256
    workers = 8
    data_dir = "/mnt/h/dataset/imagenet100"  # dataset path

    print(f"=> Custom {data_dir} is used!")
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        sampler=None,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        sampler=None,
        drop_last=True
    )

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    class_count = len(classes)

    # 1. model
    model_name = "resnet18"
    model = models.__dict__[model_name]().to(device)
    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device)

    # 2. load target pretrained model
    check_path = "./checkpoints/resnet18.pth.tar"
    model.load_state_dict(torch.load(check_path, map_location=device))
    test_acc1 = test(
        val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10
    )
    print(f"acc before ptq : {test_acc1}")


    # evaluate model status
    if False:
        print(f"model: {model}")  # print model structure
        summary(
            model, (3, 224, 224)
        )  # print output shape & total parameter sizes for given input size

    # 2. train
    epochs = 10
    model_name += '_1'
    writer = SummaryWriter(f"logs/{model_name}")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = FusedAdam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.2), int(epochs * 0.6)], gamma=0.1)

    ASP.prune_trained_model(model, optimizer)

    filename = f"./checkpoints/{model_name}_pruned_0.pth.tar"
    torch.save(model.state_dict(), filename)

    print("=> Model training has started!")
    best_acc1 = 0
    for epoch in range(epochs):
        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            scaler,
            use_amp,
            writer,
            None,
            10,
        )


        # evaluate on validation set
        acc1 = validate(
            val_loader,
            model,
            criterion,
            epoch * len(train_loader),
            device,
            class_to_idx,
            classes,
            writer,
            False,
            5,
        )

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": model_name,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            model_name,
        )

        if is_best:
            filename = f"./checkpoints/{model_name}_pruned.pth.tar"
            torch.save(model.state_dict(), filename)

    writer.close()


if __name__ == "__main__":
    main()
