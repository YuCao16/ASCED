import torch
import torch.nn.functional as F


def expand_mask(mask, expand_pixels):
    mask = mask.bool().float()  # Convert to float to use with F.conv2d
    kernel_size = 2 * expand_pixels + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    padding = expand_pixels

    expanded_mask = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0), kernel, padding=padding
    )
    return (expanded_mask > 0).squeeze()


def correct_sample(
    diffusion,
    i,
    img,
    pred_x0,
    mask,
    expand_pixels=5,
    noise_ptb_scale=0.0,
):
    pred_x0 = pred_x0 + noise_ptb_scale * torch.randn_like(pred_x0)
    img_corrected = diffusion.q_sample(pred_x0, i)

    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).to(img.device)
    elif mask.device != img.device:
        mask = mask.to(img.device)

    # Expand the mask if needed
    if expand_pixels > 0:
        mask = expand_mask(mask, expand_pixels)

    mask = mask.bool()
    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(img)

    return img_corrected * mask + img * (~mask)
