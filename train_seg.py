import os
import torch
import argparse
from models.models import LightMedSeg, LMSBR
from utils.dataset import ISLESDataset
from utils.loss import LightMedSegLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from pathlib import Path
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# from test_model import visualize_prediction

from tqdm import tqdm


def calculate_deep_supervision_loss(preds_list, targets, criterion, weights):
    """
    preds_list: List of tensors [out4, out3, out2, out1] all interpolated to target size
    targets: The single full-resolution ground truth mask (128x128x128)
    criterion: Your custom loss function (e.g., LightMedSegLoss)
    """
    deep_loss = 0.0

    # We only want to track the metrics of the main, highest-res output for logging
    main_total, main_dice, main_ce, main_bdry = 0.0, 0.0, 0.0, 0.0

    for i, pred in enumerate(preds_list):
        # Calculate loss for this specific decoder block's output
        # criterion returns: (total_combined_loss, dice_loss, ce_loss, boundary_loss)
        loss, l_dice, l_ce, l_bdry = criterion(pred, targets)

        # Multiply by the decaying weight and add to the total gradient
        deep_loss += weights[i] * loss

        # Save the metrics from D4 (index 0) so your progress bar looks accurate
        if i == 0:
            main_total = loss
            main_dice = l_dice
            main_ce = l_ce
            main_bdry = l_bdry

    # Normalize the loss so the magnitude doesn't blow up your learning rate
    weight_sum = sum(weights)
    deep_loss = deep_loss / weight_sum

    return deep_loss, (main_total, main_dice, main_ce, main_bdry)


def get_losses(
    model, images, metadata, targets, criterion, model_type, deep_supervision
):
    if model_type == "base":
        if deep_supervision:
            preds = model.forward_deep(images, metadata)
        else:
            logits = model.forward(images, metadata)
            preds = [logits]
    else:
        if deep_supervision:
            preds = model.forward_deep(images, metadata)
        else:
            preds = model.forward_train(images, metadata)

    return calculate_deep_supervision_loss(
        preds, targets, criterion, [2.0, 1.0, 0.5, 0.25, 0.125]
    )


def train_model(
    model: LightMedSeg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type,
    num_epochs=100,
    # learning rate range initial (max) to final (min)
    lr=(2e-4, 1e-9),
    deep_supervision=False,
    ce_only=False,
    device="cuda",
    last_epoch=-1,
    best_val_loss = float('inf'),
    optimizer_state_dict=None,
    save_path_best="lightmedseg_best.pth",
    save_path_last="lightmedseg_last.pth",
):

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr[0], weight_decay=0)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    criterion = LightMedSegLoss(num_classes=2, ce_only=ce_only)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=lr[1], T_max=num_epochs, last_epoch=last_epoch
    )

    best_val_loss = best_val_loss

    for epoch in range(last_epoch + 1, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n")

        # ==============
        # Training phase
        # ==============
        model.train()

        train_loss, train_dice, train_ce, train_bdry = 0.0, 0.0, 0.0, 0.0

        train_loop = tqdm(train_loader, desc="Train")

        for batch in train_loop:

            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            targets = batch["mask"].to(device)

            # print("BATCH SHAPE:", images.shape)

            optimizer.zero_grad(set_to_none=True)

            deep_loss, (loss, l_dice, l_ce, l_bdry) = get_losses(
                model,
                images,
                metadata,
                targets,
                criterion,
                model_type,
                deep_supervision,
            )

            deep_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += l_dice.item()
            train_ce += l_ce.item()
            train_bdry += l_bdry.item()

            free_mem, total_mem = torch.cuda.mem_get_info()

            used_mem = (total_mem - free_mem) / 2**20
            total_mem = total_mem / 2**20

            train_loop.set_postfix(
                Tot=f"{loss.item():.3f}",
                Dice=f"{l_dice.item():.3f}",
                CE=f"{l_ce.item():.3f}",
                Bdry=f"{l_bdry.item():.3f}",
                Mem=f"{used_mem}MiB/{total_mem}MiB",
            )

        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches
        scheduler.step()

        # ================
        # Validation phase
        # ================

        model.eval()
        val_loss, val_dice, val_ce, val_bdry = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Val")

            for batch in val_loop:
                images = batch["image"].to(device)
                metadata = batch["metadata"].to(device)
                targets = batch["mask"].to(device)

                with autocast(device_type=device, dtype=torch.float32):
                    _, (loss, l_dice, l_ce, l_bdry) = get_losses(
                        model,
                        images,
                        metadata,
                        targets,
                        criterion,
                        model_type,
                        deep_supervision,
                    )

                val_loss += loss.item()
                val_dice += l_dice.item()
                val_ce += l_ce.item()
                val_bdry += l_bdry.item()

                val_loop.set_postfix(
                    Tot=f"{loss.item():.3f}",
                    Dice=f"{l_dice.item():.3f}",
                    CE=f"{l_ce.item():.3f}",
                    Bdry=f"{l_bdry.item():.3f}",
                )

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Train | Tot: {avg_train_loss:.4f}  Dice: {train_dice/num_train_batches:.4f}  "
            f"CE: {train_ce/num_train_batches:.4f}  Bdry: {train_bdry/num_train_batches:.4f}"
        )
        print(
            f"Val   | Tot: {avg_val_loss:.4f}  Dice: {val_dice/num_val_batches:.4f}  "
            f"CE: {val_ce/num_val_batches:.4f}  Bdry: {val_bdry/num_val_batches:.4f} | LR: {current_lr:.2e}"
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

    parser.add_argument(
        "-s",
        "--model-size",
        help="Model size: 'small', 'medium' or 'large'.",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Whether to use the base model or the one with boundary refinement.",
        default="base",
        choices=["base", "refined"],
    )
    parser.add_argument(
        "--deep-supervision", help="Use deep supervision.", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--ignore-metadata",
        help="Disables the metadata FiLM functionality.",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--crop", help="Train using random crop.", action="store_true"
    )
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

    parsed["output_dir"] = args.output

    split = args.lr_range.split(":")
    parsed["lr_range"] = (float(split[0]), float(split[1]))

    parsed["model"] = args.model
    parsed["model_size"] = args.model_size
    parsed["epochs"] = args.epochs
    parsed["batch_size"] = args.batch_size
    parsed["crop"] = args.crop
    parsed["domain_augment"] = args.domain_augment
    parsed["deep_supervision"] = args.deep_supervision
    parsed["metadata_film"] = not args.ignore_metadata
    parsed["downsample"] = not args.crop
    parsed["resume"] = args.resume
    rng = args.range
    parsed["data_range"] = None if rng == None else [int(idx) for idx in rng.split(":")]

    return parsed


def main():
    torch.manual_seed(42)

    args = parse_arguments()

    output_dir = args["output_dir"]
    lr_range = args["lr_range"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    crop = args["crop"]
    domain_augment = args["domain_augment"]
    deep_supervision = args["deep_supervision"]
    metadata_film = args["metadata_film"]
    downsample = args["downsample"]
    resume = args["resume"]
    data_range = args["data_range"]
    print(data_range)

    dataset = ISLESDataset(
        split="train", range=data_range, random_crop=crop, domain_augment=domain_augment
    )

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if resume:
        last_weights_path = os.path.join(output_dir, "last.pth")
        best_weights_path = os.path.join(output_dir, "best.pth")

        checkpoint = torch.load(last_weights_path)
        model_type = checkpoint["model"]
        if model_type == "base":
            model = LightMedSeg.load(last_weights_path)
        else:
            model = LMSBR.load(last_weights_path)

        last_epoch = checkpoint["metadata"]["epoch"]
        best_val_loss = torch.load(best_weights_path)["metadata"]["val_loss"]
        optimizer_state_dict = checkpoint["metadata"]["optimizer_state_dict"]
    else:
        last_epoch = -1
        best_val_loss = float('inf')
        optimizer_state_dict = None

        if args["model"] == "base":
            if args["model_size"] == "small":
                model = LightMedSeg.small(
                    n_classes=2,
                    in_channels=1,
                    metadata_film=metadata_film,
                    downsample=downsample,
                )
            elif args["model_size"] == "medium":
                model = LightMedSeg.medium(
                    n_classes=2,
                    in_channels=1,
                    metadata_film=metadata_film,
                    downsample=downsample,
                )
            else:
                model = LightMedSeg.large(
                    n_classes=2,
                    in_channels=1,
                    metadata_film=metadata_film,
                    downsample=downsample,
                )
        else:
            if args["model_size"] == "small":
                model = LMSBR.small(
                    n_classes=2,
                    metadata_film=metadata_film,
                )
            elif args["model_size"] == "medium":
                model = LMSBR.medium(
                    n_classes=2,
                    metadata_film=metadata_film,
                )
            else:
                model = LMSBR.large(
                    n_classes=2,
                    metadata_film=metadata_film,
                )

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # out_dir = Path(output_dir)

    # for part in

    os.makedirs(output_dir, exist_ok=True)

    train_model(
        model,
        train_loader,
        val_loader,
        model_type=args["model"],
        num_epochs=epochs,
        device=device,
        lr=lr_range,
        deep_supervision=deep_supervision,
        last_epoch=last_epoch,
        optimizer_state_dict=optimizer_state_dict,
        best_val_loss=best_val_loss,
        # ce_only=True,
        save_path_best=os.path.join(output_dir, "best.pth"),
        save_path_last=os.path.join(output_dir, "last.pth"),
    )

    # image = dataset[0]["image"].to(device)
    # mask = dataset[0]["mask"].to(device)

    # visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()
