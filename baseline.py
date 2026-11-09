
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# Paths
ROOT = Path.cwd()
DATA_DIR = ROOT / "FracAtlas"
PROCESSED_DIR = ROOT / "processed"

SPLIT_DIR = PROCESSED_DIR / "splits"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
RESULTS_DIR = PROCESSED_DIR / "results"
SUBSET_DIR = PROCESSED_DIR / "labeled_subsets"

for directory in [CHECKPOINT_DIR, RESULTS_DIR, SUBSET_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Training configuration
IMAGE_SIZE = 512
BATCH_SIZE = 2
MAX_EPOCHS = 20
PATIENCE = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_FRACTIONS = [1.0, 0.30, 0.20, 0.10]

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


def load_data():
    """Load dataset metadata, annotations, and saved splits."""

    # Load COCO annotations
    coco_path = (
        DATA_DIR / "Annotations" / "COCO JSON"
        / "COCO_fracture_masks.json"
    )

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_images = {
        img["file_name"]: img
        for img in coco["images"]
    }

    annotations_by_image = {}

    for ann in coco["annotations"]:
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    # Index image files
    image_paths = {}

    image_dir = DATA_DIR / "images"

    for folder_name in ["Fractured", "Non_fractured"]:
        folder = image_dir / folder_name

        for path in folder.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and path.name != "IMG0002628.jpg"
            ):
                image_paths[path.name] = path

    # Load saved splits
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("Indexed images:", len(image_paths))

    return train_df, val_df, test_df, image_paths, coco_images, annotations_by_image


class FracAtlasDataset(Dataset):
    def __init__(self, dataframe, image_paths, coco_images, annotations_by_image, image_size=512):
        self.df = dataframe.reset_index(drop=True)
        self.image_paths = image_paths
        self.coco_images = coco_images
        self.annotations_by_image = annotations_by_image
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]

        image_path = self.image_paths[filename]

        with Image.open(image_path) as img:
            image = img.convert("L")
            original_width, original_height = image.size

            mask = Image.new(
                "L",
                (original_width, original_height),
                0,
            )

            draw = ImageDraw.Draw(mask)

            info = self.coco_images.get(filename)

            if info is not None:
                if (info["width"], info["height"]) != (
                    original_width, original_height
                ):
                    raise ValueError(
                        f"Image/COCO dimension mismatch: {filename}"
                    )

                for ann in self.annotations_by_image.get(
                    info["id"], []
                ):
                    for polygon in ann.get("segmentation", []):
                        if len(polygon) >= 6 and len(polygon) % 2 == 0:
                            points = list(zip(
                                polygon[0::2],
                                polygon[1::2],
                            ))
                            draw.polygon(points, fill=255)

            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )

            mask = mask.resize(
                (self.image_size, self.image_size),
                Image.Resampling.NEAREST,
            )

        image_array = np.array(
            image, dtype=np.float32
        ) / 255.0

        mask_array = (
            np.array(mask) > 0
        ).astype(np.float32)

        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)

        return image_tensor, mask_tensor

def generate_labeled_subsets(
    train_df,
    fractions=(0.10, 0.20, 0.30, 1.00),
    seed=42,
    force_regenerate=False,
):
    """
    Generate reproducible, nested, stratified labeled subsets
    and save them as CSV files.
    """

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    expected_files = [
        SUBSET_DIR / f"train_{int(f * 100)}pct.csv"
        for f in fractions
    ]

    # Reuse existing subsets if all are available
    if not force_regenerate and all(
        path.exists() for path in expected_files
    ):
        print("Labeled subsets already exist. Skipping generation.")
        return

    rng = np.random.default_rng(seed)

    # Shuffle fracture-positive and negative images separately
    class_groups = {}

    for label in sorted(train_df["fractured"].unique()):
        group = train_df[
            train_df["fractured"] == label
        ].copy()

        shuffled_indices = rng.permutation(len(group))
        class_groups[label] = group.iloc[shuffled_indices]

    # Generate each nested subset
    for fraction in sorted(fractions):
        selected_groups = []

        for label, group in class_groups.items():
            if fraction == 1.0:
                selected = group.copy()
            else:
                count = round(len(group) * fraction)
                selected = group.iloc[:count].copy()

            selected_groups.append(selected)

        subset = pd.concat(
            selected_groups,
            ignore_index=True,
        )

        subset = subset.sample(
            frac=1,
            random_state=seed,
        ).reset_index(drop=True)

        percentage = int(fraction * 100)

        output_path = (
            SUBSET_DIR / f"train_{percentage}pct.csv"
        )

        subset.to_csv(output_path, index=False)

        print(
            f"Saved {percentage}% subset: "
            f"{len(subset)} images, "
            f"{int(subset['fractured'].sum())} positive images"
        )


def load_labeled_subsets(
    fractions=(0.10, 0.20, 0.30, 1.00),
):
    """Load previously generated labeled subsets."""

    subsets = {}

    for fraction in fractions:
        percentage = int(fraction * 100)

        path = SUBSET_DIR / f"train_{percentage}pct.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing labeled subset: {path}. "
                "Run generate_labeled_subsets() first."
            )

        subsets[percentage] = pd.read_csv(path)

    return subsets


def create_loaders(train_subset, val_df, test_df, image_paths, coco_images, annotations_by_image, batch_size=2):
    train_dataset = FracAtlasDataset(
        train_subset,
        image_paths,
        coco_images,
        annotations_by_image,
        IMAGE_SIZE,
    )

    val_dataset = FracAtlasDataset(
        val_df,
        image_paths,
        coco_images,
        annotations_by_image,
        IMAGE_SIZE,
    )

    test_dataset = FracAtlasDataset(
        test_df,
        image_paths,
        coco_images,
        annotations_by_image,
        IMAGE_SIZE,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def create_model():
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits.float())
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(1, 2, 3))

    denominator = (
        probs.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + eps) / (denominator + eps)

    return 1 - dice.mean()


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    )

    return bce + dice_loss(logits, targets)


def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def create_scaler():
    return torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )


def calculate_metrics(logits, masks, threshold=0.5, eps=1e-7):
    predictions = (
        torch.sigmoid(logits.float()) >= threshold
    ).float()

    masks = masks.float()
    dims = (1, 2, 3)

    intersection = (predictions * masks).sum(dim=dims)
    pred_sum = predictions.sum(dim=dims)
    target_sum = masks.sum(dim=dims)

    dice = (2 * intersection + eps) / (
        pred_sum + target_sum + eps
    )

    iou = (intersection + eps) / (
        pred_sum + target_sum - intersection + eps
    )

    positive = target_sum > 0

    tp = (predictions * masks).sum()
    fp = (predictions * (1 - masks)).sum()
    fn = ((1 - predictions) * masks).sum()

    return {
        "dice": dice,
        "iou": iou,
        "positive_dice": dice[positive],
        "positive_iou": iou[positive],
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
    }


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, masks in tqdm(
        loader, desc="Training", leave=False
    ):
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = combined_loss(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_dice = []
    all_iou = []
    positive_dice = []
    positive_iou = []

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for images, masks in tqdm(
        loader, desc="Validation", leave=False
    ):
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = combined_loss(logits, masks)

        metrics = calculate_metrics(
            logits,
            masks,
            threshold,
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_dice.extend(metrics["dice"].cpu().tolist())
        all_iou.extend(metrics["iou"].cpu().tolist())

        positive_dice.extend(
            metrics["positive_dice"].cpu().tolist()
        )
        positive_iou.extend(
            metrics["positive_iou"].cpu().tolist()
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

    eps = 1e-7

    return {
        "loss": total_loss / max(total_samples, 1),
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),

        "positive_dice": (
            float(np.mean(positive_dice))
            if positive_dice else float("nan")
        ),
        "positive_iou": (
            float(np.mean(positive_iou))
            if positive_iou else float("nan")
        ),

        "precision": (total_tp + eps) / (
            total_tp + total_fp + eps
        ),
        "recall": (total_tp + eps) / (
            total_tp + total_fn + eps
        ),
    }


def train(percentage, train_loader, val_loader, max_epochs=30, patience=5):
    # Reset random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Create a fresh model and optimizer
    model = create_model()
    optimizer = create_optimizer(model)
    scaler = create_scaler()

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"unet_resnet18_{percentage}pct_best.pth"
    )

    history_path = (
        RESULTS_DIR
        / f"baseline_{percentage}pct_history.csv"
    )

    history = []
    best_val_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
        )

        val_metrics = validate(
            model,
            val_loader,
        )

        epoch_time = time.time() - start

        current_dice = val_metrics["positive_dice"]

        row = {
            "labeled_percentage": percentage,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_positive_dice": current_dice,
            "val_iou": val_metrics["iou"],
            "val_positive_iou": val_metrics["positive_iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "seconds": epoch_time,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Positive Dice: {current_dice:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"{epoch_time:.1f}s"
        )

        # Save best model based on positive-image Dice
        if (
            np.isfinite(current_dice)
            and current_dice > best_val_dice
        ):
            best_val_dice = current_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "labeled_percentage": percentage,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_positive_dice": best_val_dice,
                },
                checkpoint_path,
            )

            print("Saved best checkpoint.")

        else:
            epochs_without_improvement += 1

        # Save history after every epoch
        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after {patience} "
                "epochs without improvement."
            )
            break

    print(
        f"\n{percentage}% experiment complete. "
        f"Best validation positive Dice: {best_val_dice:.4f}"
    )

    return {
        "percentage": percentage,
        "best_val_positive_dice": best_val_dice,
        "checkpoint": checkpoint_path,
        "history": history_path,
    }
