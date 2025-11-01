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
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
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
    """
    Read user-defined layer-wise target pruning ratios from YAML and
    verify that the layer names exist in the model AND are prunable
    (i.e. conv-like params, not BN, not bias, not classifier).

    HW constraints:
    - Only CONV layers need to be pruned / counted.

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
        raise TypeError("[YAML ERROR] filename must be a str")

    # 2) Open YAML file for reading
    with open(file_name, "r") as stream:
        raw_dict = yaml.safe_load(stream)
        prune_ratio_dict = raw_dict["prune_ratios"]

    # 3) Get Param Names
    model_params = {name: p for name, p in model.named_parameters()}

    # 4) Validated dict
    validated: dict[str, float] = {}

    # 5) Check each layer and ratio
    for layer_name, ratio in prune_ratio_dict.items():
        # 5.1) must exist
        if layer_name not in model_params:
            raise ValueError(f"[YAML ERROR] {layer_name} not found in model params.")

        # 5.2) Get Param
        p = model_params[layer_name]

        # 5.3) Must be a conv-like weight (4D); skip BN/bias/FC here
        if p.dim() != 4:
            continue

        # 5.4) ratio must be in [0, 1]
        r = float(ratio)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"[YAML ERROR] {layer_name} has invalid ratio {ratio}")

        validated[layer_name] = r

    # 6) Return validated dict
    return validated


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

    # 1) Validate sparsity
    if sparsity < 0.0:
        raise ValueError("sparsity must be >= 0.0")

    # 2) Edge case: no pruning
    if sparsity == 0.0:
        return torch.ones_like(tensor, dtype=tensor.dtype)

    # 3) Edge case: full pruning (sparsity >= 1.0)
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

            # 1.1) only layers in YAML to be pruned
            if name not in prune_ratio_dict:
                continue

            # 1.2) Only prune conv-like weights (4D). Intentionally skip BN, bias, etc.
            if p.dim() != 4:
                continue

            # 1.3) Do not prune first layer
            if name == "features.0.weight":  # do not prune first conv
                continue

            ratio = float(prune_ratio_dict[name])

            # 1.4) Build pruning mask according to selected sparsity type
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

    print("Sparsity type is: {}".format(sparsity_type))

    # 3) Iterate through all parameters
    for name, p in model.named_parameters():
        w = p.data

        if sparsity_type == "unstructured":
            # 3.1) Count zeros for any shape
            zeros = (w == 0).sum().item()
            params = w.numel()
            total_zeros += zeros
            total_params += params
            layer_sparse_pct = (zeros / params * 100.0) if params > 0 else 0.0
            print("(zero/total) weights of {} is: ({}/{}). Sparsity is: {:.2f}%".format(
                name, zeros, params, layer_sparse_pct
            ))
        else:
            # 3.2) Filter pruning only makes sense for conv weights: [OC, IC, kH, kW]
            if w.dim() != 4:
                continue
            oc = w.size(0)
            flat = w.view(oc, -1).abs().sum(dim=1)
            empty = (flat == 0).sum().item()
            total_empty += empty
            total_filters += oc
            layer_sparse_pct = (empty / oc * 100.0) if oc > 0 else 0.0
            print("(empty/total) filter of {} is: ({}/{}). filter sparsity is: {:.2f}%".format(
                name, empty, oc, layer_sparse_pct
            ))

    # 4) Print global stats
    print("-" * 75)
    if sparsity_type == "unstructured":
        overall = (total_zeros / total_params) if total_params > 0 else 0.0
        print("total number of zeros: {}, non-zeros: {}, overall sparsity is: {:.4f}".format(
            total_zeros,
            total_params - total_zeros,
            overall,
        ))
    else:
        overall = (total_empty / total_filters) if total_filters > 0 else 0.0
        print("total number of filters: {}, empty-filters: {}, overall filter sparsity is: {:.4f}".format(
            total_filters,
            total_empty,
            overall,
        ))


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


def oneshot_magnitude_prune(
    *,
    model: nn.Module,
    device: torch.device,
    train_loader,
    test_loader,
    sparsity_type: str,
    prune_ratio_dict: dict,
    base_epochs: int = 15,
):
    """One-shot pruning + masked finetuning + eval.

    Steps:
    1) Apply pruning once from YAML-defined ratios.
    2) Finetune while reapplying masks so pruned weights stay zero.
    3) Evaluate final accuracy.

    Args:
        model (nn.Module): Model to prune and retrain.
        device (torch.device): Device to run training/eval on.
        train_loader (DataLoader): Training data for masked retrain.
        test_loader (DataLoader): Test/val data for final eval.
        sparsity_type (str): "unstructured" or "filter".
        prune_ratio_dict (dict): param_name -> sparsity ratio.
        base_epochs (int): Finetune epochs after pruning.

    Returns:
        Tuple[nn.Module, dict, float]: (pruned+retrained model, masks, test_acc)
    """

    # 1) Prune once using YAML ratios; make sure apply_pruning returns masks: name -> mask tensor
    masks = apply_pruning(model, prune_ratio_dict, sparsity_type)
    test_sparsity(model, sparsity_type)

    # 2) Masked retrain with small LR + cosine
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4,)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * base_epochs, eta_min=4e-8)
    criterion = nn.CrossEntropyLoss()

    # 3) Retrain
    masked_retrain(
        model=model,
        masks=masks,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=base_epochs,
    )

    # 3) Final accuracy
    acc = test(model, device, test_loader)
    return model, masks, acc


def iterative_magnitude_prune(
    *,
    model: nn.Module,
    device: torch.device,
    train_loader,
    test_loader,
    sparsity_type: str,
    prune_ratio_dict: dict,
    base_epochs: int = 15,
) -> tuple[nn.Module, dict, float]:
    """Iterative Magnitude Pruning (IMP) with masked finetuning at each stage.

    We do pruning in T (4 in this case) stages.

    Args:
        model (nn.Module): Model to prune iteratively.
        device (torch.device): Device for training/eval.
        train_loader (DataLoader): Data for masked finetuning after each step.
        test_loader (DataLoader): Data for final eval.
        sparsity_type (str): "unstructured" or "filter".
        prune_ratio_dict (dict): param_name -> final target sparsity (0..1).
        base_epochs (int): Finetune epochs per round.

    Returns:
        Tuple[nn.Module, float]: Final pruned model and test accuracy.
    """

    # 1) Decide stages of iteration based on sparsity type
    if sparsity_type == "unstructured":
        stage_targets = [0.40, 0.55, 0.70, 0.80]
        final_target = 0.80
    elif sparsity_type == "filter":
        stage_targets = [0.20, 0.28, 0.35, 0.40]
        final_target = 0.40
    else:
        raise ValueError(f"unsupported sparsity_type={sparsity_type}")

    # 2) Iter through stages
    last_masks: dict[str, torch.Tensor] = {}
    stage_epochs = [15, 12, 12, 15]
    stage_lrs = [0.02, 0.02, 0.01, 0.01]
    for stage_idx, (stage_target, epochs, lr) in enumerate(
            zip(stage_targets, stage_epochs, stage_lrs), start=1
    ):
        # 1.1) build stage-specific ratios
        stage_prune_dict = {}
        scale = stage_target / final_target
        for name, yaml_ratio in prune_ratio_dict.items():
            r = float(yaml_ratio) * scale
            if r < 0.0:
                r = 0.0
            if r > 1.0:
                r = 1.0
            stage_prune_dict[name] = r


        # 1.2) apply pruning for this stage
        masks = apply_pruning(model, stage_prune_dict, sparsity_type)
        test_sparsity(model, sparsity_type)

        # 1.3) Masked retrain with small LR + cosine
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=4e-8)
        criterion = nn.CrossEntropyLoss()

        # 1.4) Retrain
        masked_retrain(
            model=model,
            masks=masks,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=epochs,
        )
        print(f"Stage {stage_idx}/{len(stage_targets)}") # remove later

        # 1.5) Evaluate
        acc = test(model, device, test_loader)
        last_masks = masks

    # 2) Final accuracy
    acc = test(model, device, test_loader)
    return model, last_masks, acc


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

    # 4) Read prune ratio from yaml
    prune_dict = read_prune_ratios_from_yaml(args.yaml_path, model)

    # 5) Run OMP or IMP
    if args.sparsity_method == "omp":
        model, masks, acc = oneshot_magnitude_prune(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            sparsity_type=args.sparsity_type,
            prune_ratio_dict=prune_dict,
            base_epochs=15,
        )
    elif args.sparsity_method == "imp":
        model, masks, acc = iterative_magnitude_prune(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            sparsity_type=args.sparsity_type,
            prune_ratio_dict=prune_dict,
            base_epochs=15,
        )
    else:
        raise ValueError(f"unrecognized sparsity method: {args.sparsity_method}")

    # 8) Save model
    target = "0.80" if args.sparsity_type == "unstructured" else "0.40"
    fname = f"{args.sparsity_method}_{args.sparsity_type}_{target}_acc_{acc:.3f}.pt"
    torch.save(model.state_dict(), fname)
    print(f"[INFO] saved {fname}")


if __name__ == '__main__':
    main()