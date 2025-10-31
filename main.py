"""
file: main.py

Implementation file for pruning
"""

from __future__ import print_function
import os
import sys
import logging
import argparse
import time
from time import strftime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import yaml

from vgg_cifar import vgg13

# settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='training batch size (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load-model-path', type=str, default="./cifar10_vgg16_pretrained.pt",
                    help='Path to pretrained model')
parser.add_argument('--sparsity-type', type=str, default='unstructured',
                    help="define sparsity_type: [unstructured, filter, etc.]")
parser.add_argument('--sparsity-method', type=str, default='omp',
                    help="define sparsity_method: [omp, imp, etc.]")
parser.add_argument('--yaml-path', type=str, default="./vgg13.yaml",
                    help='Path to yaml file')

args = parser.parse_args()

# --- for dubeg use ---------
# args_list = [
#     "--epochs", "160",
#     "--seed", "123",
#     # ... add other arguments and their values ...
# ]
# args = parser.parse_args(args_list)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def get_dataloaders(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, download=True,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=256, shuffle=False)

    return train_loader, test_loader


# ============= the functions that you need to complete start from here =============

def read_prune_ratios_from_yaml(file_name, model):
    """Load and validate layer-wise prune ratios from a YAML file.

    This reads a YAML config that is expected to contain a top-level key
    `prune_ratios`, where each key is a parameter name in the model and each
    value is a pruning ratio between 0.0 and 1.0 (inclusive). It validates that
    all keys exist in the model and that all ratios are within range.

    Args:
        file_name (str): Path to the YAML configuration file.
        model (torch.nn.Module): Model whose parameter names are used to
            validate the YAML entries.

    Returns:
        dict[str, float]: Mapping of parameter name to pruning ratio.

    Raises:
        TypeError: If `file_name` is not a string.
        ValueError: If a layer in YAML does not exist in the model, or if a
            pruning ratio is not in [0.0, 1.0].
        yaml.YAMLError: If the YAML file cannot be parsed.
    """

    # 1) Basic type check for file path
    if not isinstance(file_name, str):
        raise TypeError("filename must be a str")

    # 2) Open YAML file for reading
    with open(file_name, "r") as stream:
        try:
            # 2.1) Load raw YAML as Python dict
            raw_dict = yaml.safe_load(stream)

            # 2.2) Extract the prune section that should exist
            prune_ratio_dict = raw_dict["prune_ratios"]

            # 3) Collect all real parameter names from the model for validation
            model_param_names = set(name for name, _ in model.named_parameters())

            # 4) Validate every YAML entry
            for layer_name, ratio in prune_ratio_dict.items():
                # 4.1) Check that layer exists in the model
                if layer_name not in model_param_names:
                    raise ValueError(
                        f"[YAML ERROR] {layer_name} not found in model params. "
                        f"Available examples: {[n for n in list(model_param_names)[:15]]}"
                    )

                # 4.2) Check that ratio is a valid float in [0.0, 1.0]
                if not (0.0 <= float(ratio) <= 1.0):
                    raise ValueError(
                        f"[YAML ERROR] {layer_name} has invalid ratio {ratio}"
                    )

            # 5) If all checks pass, return the validated dict
            return prune_ratio_dict

        except yaml.YAMLError as exc:
            # 6) Surface YAML parsing issues to the caller/log
            raise exc


def unstructured_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Create an unstructured magnitude-based pruning mask for a weight tensor.

    This does NOT change the original tensor - it only returns a mask of the
    same shape where 1 means keep and 0 means prune. The caller can then apply:
    `pruned_weight = tensor * mask`.

    Args:
        tensor (torch.Tensor): Weight tensor from a conv/linear layer.
        sparsity (float): Target fraction in [0.0, 1.0] of weights to prune
            (e.g. 0.3 means prune 30 percent smallest-magnitude weights).

    Returns:
        torch.Tensor: Mask tensor with same shape as `tensor`, dtype same as
        `tensor` (0.0 for pruned elements, 1.0 for kept elements).

    Raises:
        ValueError: If `sparsity` is negative.
    """

    # 1) Validate sparsity
    if sparsity < 0.0:
        raise ValueError("sparsity must be >= 0.0")

    # 2) Edge case: no pruning
    if sparsity == 0.0:
        return torch.ones_like(tensor, dtype=tensor.dtype)

    # 3) Edge case: full pruning (sparsity >= 1.0)
    if sparsity >= 1.0:
        return torch.zeros_like(tensor, dtype=tensor.dtype)

    # 4) Flatten magnitudes to pick global threshold; detach to avoid autograd tracking
    flat_abs = tensor.detach().abs().view(-1)
    numel = flat_abs.numel()

    # 5) How many to prune = smallest k magnitudes; If k becomes 0 due to rounding, just keep all
    k = int(numel * sparsity)
    if k == 0:
        return torch.ones_like(tensor, dtype=tensor.dtype)

    # 6) Find magnitude threshold = kth smallest magnitude; torch.kthvalue is 1-based, so we use k (already >=1 here)
    th, _ = torch.kthvalue(flat_abs, k)

    # 7) Build mask
    # Prune (set to 0) everything whose magnitude is <= threshold
    # Keep (set to 1) everything whose magnitude is > threshold
    # We use <= to hit the target sparsity more reliably
    mask = (tensor.abs() > th).to(dtype=tensor.dtype)

    # 8) Return mask for caller to apply: weight_pruned = tensor * mask
    return mask


def filter_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Create an L2-norm based filter-pruning mask for a layer weight tensor.

    For conv weights shaped [out_channels, in_channels, kH, kW], we:
    - compute L2 norm per output filter
    - prune the smallest `sparsity * out_channels` filters
    - return a mask broadcast to the full weight shape

    For linear weights shaped [out_features, in_features], we do the same but
    treat each out_feature as a "filter".

    Args:
        tensor (torch.Tensor): Layer weights. Expected 4D conv weights or 2D linear weights.
        sparsity (float): Target fraction in [0.0, 1.0] of filters to prune.

    Returns:
        torch.Tensor: Mask with same shape as `tensor`, 1.0 where kept, 0.0 where pruned.

    Raises:
        ValueError: If `sparsity` is negative.
    """

    # 1) Basic validation
    if sparsity < 0.0:
        raise ValueError("sparsity must be >= 0.0")

    # 2) Edge cases
    if sparsity == 0.0:
        return torch.ones_like(tensor, dtype=tensor.dtype)
    if sparsity >= 1.0:
        return torch.zeros_like(tensor, dtype=tensor.dtype)

    # 3) Get number of filters/output channels
    # conv: [OC, IC, kH, kW]
    # linear: [OC, IC]
    out_channels = tensor.size(0)
    num_prune = int(out_channels * sparsity)

    # 4) If rounding made it 0, just keep all; If we ended up pruning everything, return zeros
    if num_prune == 0:
        return torch.ones_like(tensor, dtype=tensor.dtype)
    if num_prune >= out_channels:
        return torch.zeros_like(tensor, dtype=tensor.dtype)

    # 5) Compute L2 norm per filter
    if tensor.dim() == 4:
        # 5.1) conv case; norm_i = ||W_i||_2 for each output filter
        filt_norms = tensor.detach().pow(2).sum(dim=(1, 2, 3)).sqrt()  # [OC]
    elif tensor.dim() == 2:
        # 5.2) linear case
        filt_norms = tensor.detach().pow(2).sum(dim=1).sqrt()  # [OC]
    else:
        # 5.3) unsupported shape
        raise ValueError(f"filter_prune expects 2D or 4D tensor, got {tensor.dim()}D")

    # 6) Threshold = kth smallest norm
    th, _ = torch.kthvalue(filt_norms, num_prune)

    # 7) Build keep vector; prune norms <= threshold
    keep_vec = (filt_norms > th).to(dtype=tensor.dtype)  # [OC]

    # 8) Broadcast to full weight shape
    if tensor.dim() == 4:
        mask = keep_vec.view(out_channels, 1, 1, 1).expand_as(tensor)
    else:  # 2D
        mask = keep_vec.view(out_channels, 1).expand_as(tensor)

    # 9) Caller can do: pruned = tensor * mask
    return mask


def apply_pruning(model: nn.Module, prune_ratio_dict: dict, sparsity_type: str) -> dict[str, torch.Tensor]:
    """Apply in-place pruning to selected conv layers of a model.

    Notes:
    - looks up each parameter name in `prune_ratio_dict`
    - skips anything not listed
    - skips non-4D params (so we ignore bias, BN, embedding, etc.)
    - builds a mask using either unstructured or filter pruning
    - multiplies the weight by the mask in-place

    Args:
        model (nn.Module): Model whose parameters will be pruned in-place.
        prune_ratio_dict (dict): Mapping from parameter name to pruning ratio
            (float in [0.0, 1.0]). Names must match `model.named_parameters()`.
        sparsity_type (str): Either "unstructured" or "filter".

    Returns:
        dict[str, torch.Tensor]: Mapping from parameter name to pruned weights.

    Raises:
        ValueError: If `sparsity_type` is not recognized.
    """

    masks: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        # 1) Walk every parameter in the model
        for name, p in model.named_parameters():
            if name not in prune_ratio_dict:
                continue

            # 1.2) Only prune conv-like weights (4D). Intentionally skip BN, bias, etc.
            if p.dim() != 4:
                continue

            ratio = float(prune_ratio_dict[name])

            # 3) Build pruning mask according to selected sparsity type
            if sparsity_type == "unstructured":
                mask = unstructured_prune(p.data, ratio)
            elif sparsity_type == "filter":
                mask = filter_prune(p.data, ratio)
            else:
                raise ValueError(f"unknown sparsity_type={sparsity_type}")

            # 4) In-place apply mask: weight_pruned = weight * mask
            p.data *= mask
            masks[name] = mask
    return masks


def test_sparsity(model: torch.nn.Module, sparsity_type: str = "unstructured") -> None:
    """Print per-layer and overall sparsity statistics for a model.

    Supports:
    - unstructured: count zero elements per parameter tensor
    - filter: count zeroed-out conv filters (OC-level) only

    Args:
        model (torch.nn.Module): Model to inspect.
        sparsity_type (str): Either "unstructured" or "filter".

    Raises:
        ValueError: If `sparsity_type` is not supported.
    """

    # 1) Validate mode
    if sparsity_type not in ("unstructured", "filter"):
        raise ValueError(f"Unsupported sparsity_type={sparsity_type}")

    # 2) Global accumulators
    total_zeros = 0       # for unstructured
    total_params = 0
    total_empty = 0       # for filter
    total_filters = 0

    print(f"Sparsity type is: {sparsity_type}")

    # 3) Iterate through all parameters
    for name, p in model.named_parameters():
        w = p.data

        if sparsity_type == "unstructured":
            # 3.1) Count zeros for any shape
            zeros = (w == 0).sum().item()
            params = w.numel()
            total_zeros += zeros
            total_params += params
            print(f"(zero/total) weights of {name} is: ({zeros}/{params}). Sparsity is: {zeros/params:.4f}")
        else:
            # 3.2) Filter pruning only makes sense for conv weights: [OC, IC, kH, kW]
            oc = w.size(0)
            flat = w.view(oc, -1).abs().sum(dim=1)
            empty = (flat == 0).sum().item()
            total_empty += empty
            total_filters += oc
            print(f"(empty/total) filter of {name} is: ({empty}/{oc}). filter sparsity is: {empty / oc:.4f}")

    # 4) Print global stats
    print("-" * 75)
    if sparsity_type == "unstructured":
        overall = total_zeros / total_params if total_params else 0.0
        print(f"total number of zeros: {total_zeros}, non-zeros: {total_params - total_zeros}, overall sparsity is: {overall:.4f}")
    else:
        overall = total_empty / total_filters if total_filters else 0.0
        print(f"total number of filters: {total_filters}, empty-filters: {total_empty}, overall filter sparsity is: {overall:.4f}")


def masked_retrain(
    *,
    model: nn.Module,
    masks: dict,
    train_loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler=None,
    epochs: int = 15,
) -> None:
    """Finetune a pruned model while keeping previously pruned weights zero.

    After each optimizer step, this re-applies the stored masks so that
    gradients cannot revive pruned weights.

    Args:
        model (nn.Module): Pruned model to retrain.
        masks (dict[str, torch.Tensor]): Dict mapping param name -> 0/1 mask
            with same shape as the parameter.
        train_loader (DataLoader): Training data loader.
        device (torch.device): Device to run training on.
        optimizer (torch.optim.Optimizer): Optimizer for model params.
        criterion (torch.nn.Module): Criterion for model params.
        scheduler (optional): LR scheduler. Called once per batch if provided.
        epochs (int): Number of training epochs.

    Notes:
        - This applies scheduler.step() per batch, not per epoch.
        - Only parameters whose names appear in `masks` are forced back to zero.
    """

    model.train()

    # remove this line later
    from tqdm import tqdm

    # 1) Loop over epochs
    for epoch in range(epochs):
        # 1.1) Loop over mini-batches
        train_loader = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", unit="batch")
        for images, targets in train_loader:
            # 2) Move batch to device
            images = images.to(device)
            targets = targets.to(device)

            # 3) Forward + loss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            # 4) Backward + update
            loss.backward()
            optimizer.step()

            # 5) Re-apply masks to keep pruned weights at 0
            # 5.1) Only touch params that have a corresponding mask
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in masks:
                        p.data *= masks[name]

            # 6) Optional LR schedule per step
            if scheduler is not None:
                scheduler.step()

            # Remove this line later
            train_loader.set_postfix(loss=loss.item())

def oneshot_magnitude_prune(model, sparity_type, prune_ratio_dict):
    # Implement the function that conducting oneshot magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # This function should also include the masked_retrain() function to conduct fine-tuning to restore the accuracy
    pass

def iterative_magnitude_prune():
    # Implement the function that conducting iterative magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # You can choose the way to gradually increase the pruning ratio.
    # For example, if the overall target sparsity is 80%, 
    # you can achieve it by 20%->40%->60%->80% or 50%->60%->70%->80% or something else e.g., in LTH paper.
    # At each sparsity level, you need to retrain your model. 
    # Therefore, this IMP method requires more overall training epochs than OMP.
    # ** IMP method needs to use at least 3 iterations.
    pass

def prune_channels_after_filter_prune():
    # 
    # You need to implement this function to complete the following task:
    # 1. This function takes a filter pruned and fine-tuned model as input
    # 2. Find out the indices of all pruned filters in each CONV layer
    # 3. Directly prune the corresponding channels (that has the same indices) in next CONV layer (on top of the filter-pruned model).
    #    There is no need to fine-tune this model again.
    # 4. Return the newly pruned model

    # E.g., if you prune the filter_1, filter_4, filter_7 from the i_th CONV layer,
    # Then, this function will let you prune the Channel_1, Channel_4, Channel_7, from the next CONV layer, i.e., (i+1)_th CONV layer.

    # How to use this function:
    # 1. You will apply this function on a filter-pruned model (after fine-tune/mask retraine)
    # 2. There is no need to fine-tune the model again after apply this function
    # 3. Compare the test accuracy before and after apply this function
    #   
    # E.g., 
    #       pruned_model = your pruned and fine/tuned model
    #       test_accuracy(pruned_model)
    #       new_model = prune_channels_after_filter_prune(pruned_model)
    #       test_accuracy(new_model)

    # Answer the following questions in your report:
    # 1. After apply this function (further prune the corresponding channels), what is the change in sparsity?
    # 2. Will accuray decrease, increase, or not change?
    # 3. Based on question 2, explain why?
    # 4. Can we apply this function to ResNet and get the same conclusion? Why?
    pass


def main():

    # 0) Check Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 1) setup random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2) Load PT model
    model = vgg13()
    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    if use_cuda:
        model.cuda()

    # 3) Data loaders
    train_loader, test_loader = get_dataloaders(args)

    # 4) CE Loss Function + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # 5) lr scheduler to fine-tune/mask-retrain your pruned model.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)

    # 6) Read prune ratio from yaml
    prune_dict = read_prune_ratios_from_yaml(args.yaml_path, model)

    # 7) Apply Masking
    masks = apply_pruning(model, prune_dict, args.sparsity_type)

    # 8) Test Sparsity with mask
    test_sparsity(model, args.sparsity_type)

    # 9) Retrain with mask
    masked_retrain(model=model, masks=masks, train_loader=train_loader, device=device, optimizer=optimizer, criterion=criterion, scheduler=scheduler, epochs=args.epochs)

    # 10) Sanity check to make sure model is still sparse and mask is working
    test_sparsity(model, args.sparsity_type)

    """
        main()
            |- read_prune_ratios_from_yaml()
            |- IMP() or OMP()
                |-apply_pruning()
                    |-unstructured_prune()
                    |-filter_prune()
                |-masked_retrain()
    """

    # ---- you can test your model accuracy and sparity using the following fuction ---------
    # test_sparsity()
    # test(model, device, test_loader)

    # ========================================
    


if __name__ == '__main__':
    main()