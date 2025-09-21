"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        target_dir (str): A directory for saving the model to.
        model_name (str): A filename for the saved model. (".pth" or ".pt")
    
    Example usage:
        save_model(model=model_1,
                   target_dir="artifacts",
                   model_name="model_1.pth"
        )
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_save_path = target_dir_path / model_name
    
    # Save the model satet_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
def plot_predictions(model, data_loader, device, n=6):
    images, labels = next(iter(data_loader))
    outputs = model(images.to(device))
    _, preds = outputs.max(1)

    fig, axes = plt.subplots(1, n, figsize=(12, 2))
    for i in range(n):
        axes[i].imshow(images[i][0], cmap="gray")
        axes[i].set_title(f"Pred: {preds[i].item()}, Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()