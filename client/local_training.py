import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def train_one_epoch(matcher, dataloader: DataLoader, optimizer: optim.Optimizer,
                   device: torch.device) -> float:
    matcher.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        input_dict = {
            "image0": batch["image_a"],
            "image1": batch["image_b"],
        }
        outputs = matcher.model(input_dict)

        loss = matcher.compute_loss(outputs, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        del input_dict, outputs, loss
        torch.cuda.empty_cache()

    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate_local_model(matcher, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    matcher.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            input_dict = {
                "image0": batch["image_a"],
                "image1": batch["image_b"]
            }
            outputs = matcher.model(input_dict)

            loss = matcher.compute_loss(outputs, batch)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"loss": avg_loss}

def fit_local_model(matcher, train_loader: DataLoader, val_loader: DataLoader,
                   config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = config.get("learning_rate", 1e-4)
    local_epochs = config.get("local_epochs", 1)
    weight_decay = config.get("weight_decay", 1e-4)

    trainable = [p for p in matcher.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found in the matcher. "
            "If LoRA is enabled, ensure enable_lora() was called before fit_local_model()."
        )
    optimizer = optim.Adam(trainable, lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    final_metrics = {}

    logger.info(f"Starting local training for {local_epochs} epochs on {device}")

    for epoch in range(local_epochs):
        train_loss = train_one_epoch(matcher, train_loader, optimizer, device)
        val_metrics = evaluate_local_model(matcher, val_loader, device)

        logger.info(f"Epoch {epoch+1}/{local_epochs}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}")

        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            final_metrics = val_metrics.copy()
            final_metrics['train_loss'] = train_loss

        gc.collect()
        torch.cuda.empty_cache()

    final_metrics['epochs_completed'] = local_epochs
    gc.collect()
    return matcher, final_metrics
