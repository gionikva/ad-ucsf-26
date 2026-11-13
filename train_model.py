import os
from typing import Literal
import torch
import argparse
from models.lightmedseg import LightMedSeg
from models.msr_cog import MSRCog
from utils.dataset import ADNICogDataset, ADNISegDataset
from utils.loss import LightMedSegLoss, MSRCogLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm


def train_model(
    model: MSRCog,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs=100,
    # learning rate range initial (max) to final (min)
    lr=(2e-4, 1e-9),
    device="cuda",
    last_epoch=-1,
    best_val_loss=float("inf"),
    optimizer_state_dict=None,
    save_path_best="lightmedseg_best.pth",
    save_path_last="lightmedseg_last.pth",
):

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr[0], weight_decay=0)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    criterion = MSRCogLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=lr[1], T_max=num_epochs, last_epoch=last_epoch
    )

    best_val_loss = best_val_loss

    for epoch in range(last_epoch + 1, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n")

        # ==============
        # Training phase
        # ==============
        model.train()

        train_loss, train_mse, train_int = 0.0, 0.0, 0.0

        train_loop = tqdm(train_loader, desc="Train")

        for batch in train_loop:
            images = batch["mri"].to(device)
            demographics = batch["dem"].to(device)
            cogscores = batch["cog"].to(device)

            optimizer.zero_grad(set_to_none=True)

            masks, preds = model(images, demographics)

            loss, l_mse, l_int = criterion(masks, preds, cogscores)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += l_mse.item()
            train_int = l_int.item()

            free_mem, total_mem = torch.cuda.mem_get_info()

            used_mem = (total_mem - free_mem) / 2**20
            total_mem = total_mem / 2**20

            train_loop.set_postfix(
                Tot=f"{loss.item():.3f}",
                MSE=f"{l_mse.item():.3f}",
                Int=f"{l_int.item():.3f}",
                Mem=f"{used_mem}MiB/{total_mem}MiB",
            )

        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches
        scheduler.step()

        # ================
        # Validation phase
        # ================

        model.eval()
        val_loss, val_mse, val_int = 0.0, 0.0, 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Val")

            for batch in val_loop:
                images = batch["mri"].to(device)
                demographics = batch["dem"].to(device)
                cogscores = batch["cog"].to(device)

                masks, preds = model(images, demographics)

                loss, l_mse, l_int = criterion(masks, preds, cogscores)

                val_loss += loss.item()
                val_mse += l_mse.item()
                val_int = l_int.item()

                val_loop.set_postfix(
                    Tot=f"{loss.item():.3f}",
                    MSE=f"{l_mse.item():.3f}",
                    Int=f"{l_int.item():.3f}",
                    Mem=f"{used_mem}MiB/{total_mem}MiB",
                )

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Train | Tot: {avg_train_loss:.4f}"
            f"MSE: {train_mse / num_train_batches:.4f}"
            f"INT: {train_int / num_train_batches:.4f}"
        )
        print(
            f"Val | Tot: {avg_val_loss:.4f}"
            f"MSE: {val_mse / num_val_batches:.4f}"
            f"INT: {val_int / num_val_batches:.4f}"
        )

        # ==========================
        #       CHECKPOINTING
        # ==========================

        metadata = {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(
                f"--> Validation loss improved to {best_val_loss:.4f}. Saving checkpoint!"
            )
            model.save(save_path_best, metadata)

        model.save(save_path_last, metadata)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seg-weights",
        help="Trained segmentation backbone weights.",
    )

    parser.add_argument(
        "-s",
        "--input-size",
        help="Input image dimensions as a 3-element tuple.",
        default=(192, 192, 192),
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Where to output the best and last weights.",
        default="out",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs for training/eval.",
        type=int,
        default=40,
    )
    parser.add_argument("-b", "--batch-size", help="Batch size.", type=int, default=1)

    parser.add_argument(
        "--lr-range",
        help="Learning rate range in the format max:min. Must follow Python's float format.",
        type=str,
        default="5e-4:1e-8",
    )

    parser.add_argument(
        "-r",
        "--range",
        help="Range of datapoints to train on in the format start_idx:end_idx.",
        type=str,
        default=None,
    )

    # parser.add_argument(
    #     "-d",
    #     "--downsample",
    #     help="Downsample the input features to avoid running out of memory.",
    #     action="store_true"
    # )
    # parser.add_argument(
    #     "--deep-supervision", help="Use deep supervision.", action="store_true"
    # )
    parser.add_argument(
        "-a",
        "--domain-augment",
        help="Enable domain augmentation.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--resume",
        help="Resume training from the last epoch.",
        action="store_true",
    )

    args = parser.parse_args()

    parsed = {}

    parsed["input_size"] = args.input_size
    parsed["output_dir"] = args.output
    parsed["seg_weights"] = args.seg_weights

    split = args.lr_range.split(":")
    parsed["lr_range"] = (float(split[0]), float(split[1]))
    # parsed["model_size"] = args.model_size
    parsed["epochs"] = args.epochs
    parsed["batch_size"] = args.batch_size
    parsed["domain_augment"] = args.domain_augment
    # parsed["deep_supervision"] = args.deep_supervision
    # parsed["downsample"] = args.downsample
    parsed["resume"] = args.resume
    rng = args.range
    parsed["data_range"] = None if rng == None else [int(idx) for idx in rng.split(":")]

    return parsed


def main():
    torch.manual_seed(42)

    args = parse_arguments()

    input_size = args["input_size"]
    output_dir = args["output_dir"]
    seg_weights = args["seg_weights"]
    lr_range = args["lr_range"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    domain_augment = args["domain_augment"]
    # deep_supervision = args["deep_supervision"]
    # downsample = args["downsample"]
    resume = args["resume"]
    data_range = args["data_range"]

    dataset = ADNICogDataset(
        split="train", test_ratio=0.1, range=data_range, domain_augment=domain_augment
    )

    print(len(dataset))

    val_ratio = 0.2
    train_ratio = 1.0 - val_ratio

    # Train/validation split
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if resume:
        last_weights_path = os.path.join(output_dir, "last.pth")
        best_weights_path = os.path.join(output_dir, "best.pth")

        checkpoint = torch.load(last_weights_path)
        model = LightMedSeg.load(last_weights_path)

        last_epoch = checkpoint["metadata"]["epoch"]
        best_val_loss = torch.load(best_weights_path)["metadata"]["val_loss"]
        optimizer_state_dict = checkpoint["metadata"]["optimizer_state_dict"]
    else:
        last_epoch = -1
        best_val_loss = float("inf")
        optimizer_state_dict = None

        model = MSRCog.from_seg(
            path=seg_weights, input_size=input_size, demo_size=14, out_size=1
        )

    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        device=device,
        lr=lr_range,
        last_epoch=last_epoch,
        optimizer_state_dict=optimizer_state_dict,
        best_val_loss=best_val_loss,
        # ce_only=True,
        save_path_best=os.path.join(output_dir, "best.pth"),
        save_path_last=os.path.join(output_dir, "last.pth"),
    )


if __name__ == "__main__":
    main()
