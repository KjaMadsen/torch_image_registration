import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Callable

def imregIB(
    ref_img : torch.Tensor, 
    mov_img : torch.Tensor, 
    loss_fn : Callable = F.mse_loss, 
    optimizer : str = 'sgd', 
    optimizer_kwargs : dict = {'lr':0.1}, 
    n_iter : int = 100,
    return_transform : bool = False,
    verbose : bool = True,
) -> torch.Tensor:
    """
    Performs intensity-based image registration.
    Args:
        ref_img: reference image
        mov_img: moving image
        loss_fn: loss function
        optimizer: optimizer
        n_iter: number of iterations
        return_transform: return the transformation matrix
    Returns:
        warped image
    """

    if ref_img.shape != mov_img.shape:
        raise ValueError("ref_img and mov_img must have the same shape.")
    if ref_img.dim() != 2:
        raise ValueError("ref_img and mov_img must be 2D. If you have an RGB image, try to convert it to grayscale first.")

    # initialize the transformation
    theta = torch.eye(2, 3, requires_grad=True)
    
    if optimizer == 'sgd':
        optimizer = optim.SGD([theta], **optimizer_kwargs)
    else:
        raise NotImplementedError
        
    # run the optimization
    for i in range(n_iter):
        optimizer.zero_grad()
        warped_img = affine_transform(mov_img, theta)
        loss = loss_fn(ref_img, warped_img)
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Iteration: {i}, Loss: {loss.item():.5f}")

    warped_img = warped_img.squeeze(0).detach().numpy()
    if return_transform:
        return warped_img, theta
    else:
       return warped_img


def affine_transform(img, theta):
    """
    Args:
        img: image to be transformed
        theta: affine transformation matrix
    Returns:
        transformed image
    """
    while img.dim() < 4:
        img = img.unsqueeze(0).unsqueeze(0)
    # create meshgrid
    grid = F.affine_grid(theta.unsqueeze(0), img.size())
 
    # sample transformed image
    warped_img = F.grid_sample(img, grid.to(torch.float), padding_mode='border')
    return warped_img.squeeze(0).squeeze(0)