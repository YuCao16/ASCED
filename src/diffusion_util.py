import numpy as np
import torch

from .utils import tensor2img
from .correction import correct_sample
from .detection import get_artifact_mask_args


def ddim_sample_loop_x0(
    diffusion,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
):
    """
    Save predicted x0 while generating.

    Same usage as ddim_sample_loop().
    """
    final = None
    x0_list = []
    for sample in diffusion.ddim_sample_loop_progressive(
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        eta=eta,
    ):
        final = sample
        x0_list.append(final["pred_xstart"])
    assert final is not None
    return x0_list


def ddim_sample_loop_progressive_find_and_correct(
    args,
    diffusion,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
):
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        img = noise
    else:
        img = torch.randn(*shape, device=device)
    indices = list(range(diffusion.num_timesteps))[::-1]
    normalized_scores = []

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        eta = 0.0
        if i == args.classification_t:
            eta = args.correct_eta
            # Process each sample in the batch
            for batch_id in range(shape[0]):
                image_sequence = [
                    np.array(tensor2img(normalized_score[batch_id : batch_id + 1])[0])
                    for normalized_score in normalized_scores
                ]
                filtered_mask = get_artifact_mask_args(
                    args, image_sequence[args.classification_start :]
                )
                if filtered_mask.sum() > 0:
                    _final_image = (
                        normalized_scores[-args.reverse_step][batch_id]
                        if args.reverse_step > 0
                        else img[batch_id]
                    )
                    img[batch_id : batch_id + 1] = correct_sample(
                        diffusion,
                        t[batch_id : batch_id + 1],
                        img=img[batch_id : batch_id + 1],
                        pred_x0=_final_image,
                        mask=filtered_mask,
                        expand_pixels=args.expand_mask,
                        noise_ptb_scale=args.noise_ptb_scale,
                    )
        with torch.no_grad():
            out = diffusion.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]
            normalized_scores.append(out["pred_xstart"])


def ddim_sample_loop_find_and_correct(
    args,
    diffusion,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
):
    final = None
    for sample in ddim_sample_loop_progressive_find_and_correct(
        args,
        diffusion,
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        eta=eta,
    ):
        final = sample
    return final["sample"]
