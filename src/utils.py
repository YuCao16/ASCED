from typing import Any, List, Optional, Union

import yaml
from PIL import Image

import numpy as np
import torch
from torch import autograd


def deterministic(seed: Optional[int]) -> None:
    if seed is None:
        seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_grid(
    images_list: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> Image.Image:
    # Automatically calculate rows or columns
    num_images = len(images_list)
    if rows is None and cols is not None:
        rows = (num_images + cols - 1) // cols  # Calculate rows from cols
    elif cols is None and rows is not None:
        cols = (num_images + rows - 1) // rows  # Calculate cols from rows
    elif cols is not None and rows is not None:
        assert (
            len(images_list) <= rows * cols
        ), "Number of images does not match grid size"
    else:
        raise ValueError("Either rows or cols must be specified")

    # Get the size of the images (assuming all have the same size)
    w, h = images_list[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    # Paste images into the grid
    for i, image in enumerate(images_list):
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def load_config(config_file: str) -> Any:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def tensor2img(sample: torch.Tensor) -> List[Image.Image]:
    image_processed = ((sample + 1.0) * 127.5).clamp(0, 255)
    image_processed = image_processed.cpu().permute(0, 2, 3, 1).contiguous()
    image_processed = image_processed.numpy().astype(np.uint8)
    return [Image.fromarray(image) for image in image_processed]


def img2tensor(img: Image.Image) -> torch.Tensor:
    img = img.resize((256, 256))
    img_np = np.array(img)
    img_np = (img_np - 127.5) / 127.5

    img_tensor = torch.tensor(img_np).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor


def modify_list_arguments(parser):
    for action in parser._actions:
        if isinstance(action.default, list):

            def type_func(v):
                if v is None:
                    return action.default
                else:
                    return v.split(",")

            action.nargs = "?"
            action.const = action.default
            action.type = type_func


def estimate_fisher(model: torch.nn.Module, loglikelihood: torch.Tensor):
    """
    Estimate Fisher Information for any nn.Module model.

    Args:
    model (nn.Module): The model for which to estimate Fisher Information.
    loglikelihood (torch.Tensor): A loglikelihood (e.g., loss) from a batch of data.

    Returns:
    tuple: Gradients and estimated Fisher Information for each trainable parameter.
    """
    # Initialize fisher information placeholder
    est_fisher_info = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            est_fisher_info[n] = p.detach().clone().zero_()

    params = [p for p in model.parameters() if p.requires_grad]
    loglikelihood_grads = autograd.grad(
        loglikelihood, params, retain_graph=True
    )

    # Square gradients and return
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.requires_grad and loglikelihood_grads[i] is not None:
            est_fisher_info[n] = loglikelihood_grads[i].detach() ** 2

    return loglikelihood_grads, est_fisher_info
