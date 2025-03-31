import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def select_device():
    """
    Returns the available device (CUDA if available, else CPU).
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(
        train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        num_epochs,
        output_path=os.path.join(os.getcwd(), "tmp")
):
    """
    Train a classification model on the training set, evaluate it on validation.
    Saves the best model (based on AUC) during training.

    Returns:
        best_accuracy, best_auc
    """
    # Timing
    training_start_time = time.time()

    # Initialize tracking variables
    max_epochs = num_epochs
    highest_auc = -1.0
    highest_acc = -1.0
    best_epoch = -1
    epoch_losses = []
    auc_scores = []

    # Set device for model
    device = select_device()
    model = model.to(device)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for epoch_idx in range(max_epochs):
        print(f"----------\nEpoch {epoch_idx + 1}/{max_epochs}")
        model.train()
        total_loss = 0.0
        iteration_count = 0

        # Training step
        for batch_idx, (feat_batch, lbl_batch, name_batch) in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train Epoch={epoch_idx}",
                ncols=80,
                leave=False
        ):
            iteration_count += 1
            feat_batch = feat_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            optimizer.zero_grad()
            _, prediction = model(feat_batch)

            loss_value = loss_fn(prediction, lbl_batch)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            print(f"{iteration_count}/{len(train_loader)} | train_loss: {loss_value.item():.4f}")

        avg_loss = total_loss / iteration_count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch_idx + 1} average loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        pred_accumulator = []
        true_accumulator = []

        with torch.no_grad():
            for batch_idx, (feat_batch, lbl_batch, name_batch) in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Validation Epoch={epoch_idx}",
                    ncols=80,
                    leave=False
            ):
                feat_batch = feat_batch.to(device)
                lbl_batch = lbl_batch.to(device)

                _, prediction = model(feat_batch)
                pred_accumulator.append(prediction)
                true_accumulator.append(lbl_batch)

        # Concatenate predictions and labels
        all_preds = torch.cat(pred_accumulator, dim=0)
        all_labels = torch.cat(true_accumulator, dim=0)

        # Compute probabilities for positive class
        prob_scores = torch.softmax(all_preds, dim=1).cpu().numpy()
        true_values = all_labels.cpu().numpy()

        # ROC AUC and accuracy
        current_auc = roc_auc_score(true_values, prob_scores, multi_class="ovr")
        current_acc = (all_preds.argmax(dim=1) == all_labels).sum().item() / len(all_labels)

        auc_scores.append(current_auc)

        # Check if this is the best so far
        if current_auc > highest_auc:
            highest_auc = current_auc
            highest_acc = current_acc
            best_epoch = epoch_idx + 1
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            print("Best model updated (new best AUC).")

        print(
            f"Epoch {epoch_idx + 1} | AUC: {current_auc:.4f} | ACC: {current_acc:.4f} | "
            f"Best AUC: {highest_auc:.4f} (ACC: {highest_acc:.4f} at epoch {best_epoch})"
        )

    # Training duration
    training_end_time = time.time()
    elapsed = training_end_time - training_start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Total Training Time: {int(h):02}:{int(m):02}:{s:.2f}")
    print(f"Training complete. Best AUC: {highest_auc:.4f} (Epoch {best_epoch})")

    return highest_acc, highest_auc