"""Main module."""
import logging
from collections.abc import Iterable
from typing import Callable, Literal, Optional, List, Dict, Any

from anndata import AnnData
import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot

from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import bbknn

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

class Augmentation:
    """
    A class to perform various data augmentation techniques on single-cell RNA-seq data.

    Attributes:
        x (torch.Tensor): The training batch containing single-cell RNA-seq data.
        num_cells (int): The number of cells in the batch.
        num_genes (int): The number of genes in the batch.
    """

    def __init__(self, x: torch.Tensor):
        """
        Initializes the Augmentation object with the given dataset.

        Args:
            dataset (Dict[str, Any]): The dataset to be augmented.
        """
        self.x = x
        self.num_cells, self.num_genes = self.x.shape

    
    def build_mask(self, masked_percentage: float) -> torch.Tensor:
        """
        Builds a random mask for gene selection based on the specified percentage.

        Args:
            masked_percentage (float): The percentage of genes to keep.

        Returns:
            torch.Tensor: A boolean mask for gene selection.
        """
        num_masked = int(self.num_genes * masked_percentage)
        mask = torch.cat([torch.ones(num_masked, dtype=torch.bool),
                          torch.zeros(self.num_genes - num_masked, dtype=torch.bool)], dim=0)
        mask = mask[torch.randperm(self.num_genes)]
        return mask

    
    def random_mask(self, x: torch.Tensor, mask_percentage: float = 0.8, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Applies random masking to the gene expression data.

        Args:
            x (torch.Tensor): The tensor containing gene expression data.
            mask_percentage (float): The percentage of genes to keep.

        Returns:
            torch.Tensor: The tensor with randomly masked genes.
        """
        dtype, device= x.dtype, x.device
        mask = self.build_mask(mask_percentage).unsqueeze(0).expand(self.num_cells, -1)
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()
        x[augmented_cell_idxs] *= mask[augmented_cell_idxs].to(dtype=dtype, device=device)
        return x

    

    def sparsity_based_masking(self, x: torch.Tensor, mask_probs: np.ndarray, mask_percentage: float = 0.2, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Applies distribution-based masking to the gene expression data.

        Args:
            x (torch.Tensor): The tensor containing gene expression data.
            mask_probs (np.ndarray): The densities (a.k.a 1 - sparsity) of each gene across the entire dataset.
            mask_percentage (float): The proportion of genes to keep.
            apply_prob: The probability of masking a given cell in the batch.

        Returns:
            torch.Tensor: The tensor with genes randomly masked based on their expressiveness.
        """
    

        dtype, device = x.dtype, x.device
        probs_to_stay = torch.tensor(mask_probs)
        probs_to_stay  = probs_to_stay.to(dtype=torch.float64, device=device)

        # there are too many low probabilities so a monotonous transformation f(x) = x^(1/10) will unskew things
        probs_to_stay = torch.pow(probs_to_stay, 1/10)
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()

        gene_mask = torch.bernoulli(probs_to_stay)
        gene_mask = gene_mask.to(device=device)

        num_genes_to_mask = int(self.num_genes * mask_percentage)
        num_masked_genes = torch.sum(gene_mask == 0).item()

        if num_masked_genes > num_genes_to_mask:
            masked_indices = torch.nonzero(gene_mask == 0)
            sorted_indices = torch.argsort(probs_to_stay[masked_indices], descending=True)
            num_to_unmask = num_masked_genes - num_genes_to_mask
            unmask_indices = masked_indices[sorted_indices][:num_to_unmask]
            gene_mask[unmask_indices] = 1

        gene_mask = gene_mask.unsqueeze(0).expand(self.num_cells, -1)

        masked_x = x.clone()  # Create a copy to preserve original data
        masked_x[augmented_cell_idxs] *= gene_mask[augmented_cell_idxs].to(dtype=dtype)

        return masked_x

        
    
    def random_swap(self, x: torch.Tensor, swap_percentage: float = 0.1, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Randomly swaps gene expressions between pairs of genes in the data.

        Args:
            x (torch.Tensor): The tensor containing gene expression data.
            swap_percentage (float): The proportion of genes to be swapped.
            apply_prob (float): The probability of each gene of being affected by the random swap.

        Returns:
            torch.Tensor: The tensor with swapped gene expressions.
        """
        swap_instances = int(self.num_genes * swap_percentage / 2)
        swap_indices = torch.randperm(self.num_genes)[:2 * swap_instances]
        swap_indices = swap_indices.view(swap_instances, 2)  # Reshape into pairs for proper indexing

        x_augmented = x.clone()
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()
        swap_values = x_augmented[:, swap_indices[:, 0]].clone()
        x_augmented[:, swap_indices[:, 0]] = x_augmented[:, swap_indices[:, 1]]
        x_augmented[:, swap_indices[:, 1]] = swap_values

        x_augmented[~augmented_cell_idxs] = x[~augmented_cell_idxs]

        return x_augmented


    def log_normal_noise(self, x: torch.Tensor, mean: float = 0, std: float = 0.5, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Adds log-normal noise to each entry of the gene expression data.

        Args:
            x (torch.Tensor): The tensor containing gene expression data.
            mean (float): Mean of the underlying normal distribution.
            std (float): Standard deviation of the underlying normal distribution.
            apply_prob (float): The probability of applying noise to a given cell.

        Returns:
            torch.Tensor: The tensor with added log-normal noise.
        """
        device = x.device
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()
        noise = torch.empty_like(x).log_normal_(mean=mean, std=std)
        noise = noise.to(device)
        x_augmented = x.clone()
        x_augmented[augmented_cell_idxs] += noise[augmented_cell_idxs]
        return x_augmented

    
    def poisson_noise(self, x: torch.Tensor, max_poisson_param: float = 1.0, apply_prob: float = 0.5, per_gene: bool = False) -> torch.Tensor:
        """
        Adds Poisson noise to the gene expression data.

        Args:
            x (torch.Tensor): The tensor containing gene expression data.
            max_poisson_param (float): The maximum parameter value for scaling noise rates.
            apply_prob (float): The probability of applying noise to a given cell.
            per_gene (bool): When True, we compute noise parameters per gene, otherwise we compute noise parameters per cell.

        Returns:
            torch.Tensor: The tensor with added Poisson noise.
        """

        device = x.device
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()

        if per_gene:
            dim = 0
            mean_gene_counts = torch.nanmean(x.float(), dim=dim, keepdim=True)
            noise_rates = mean_gene_counts.expand(self.num_cells, self.num_genes).to(device)
        else:
            dim = 1
            library_sizes = torch.sum(x, dim=dim, keepdim=True).to(device)
            noise_rates = library_sizes.expand(self.num_cells, self.num_genes).to(device)

        noise_rates_scaled = (noise_rates / noise_rates.max()) * max_poisson_param
        noise = torch.poisson(noise_rates_scaled)
        noise = noise.to(device)

        x_augmented = x.clone()
        x_augmented[augmented_cell_idxs] += noise[augmented_cell_idxs]

        return x_augmented



    def instance_crossover(self, x: torch.Tensor, gene_swap_prop: float = 0.2, cell_swap_prop: float = 0.2) -> torch.Tensor:
        """
        Performs a crossover operation between pairs of cells in the tensor.

        Args:
            x (torch.Tensor): The tensor of scRNA-seq data.
            gene_swap_prop (float): The proportion of genes to be crossed over.
            cell_swap_prop (float): The proportion of cells to be affected by crossover.

        Returns:
            torch.Tensor: The tensor with crossed-over gene expressions.
        """

        dtype, device = x.dtype, x.device
        num_genes_to_swap = int(self.num_genes * gene_swap_prop)
        num_cells_to_swap = int(self.num_cells * cell_swap_prop / 2)
        
        gene_swap_indices = torch.randperm(self.num_genes)[:num_genes_to_swap]
        cell_swap_indices = torch.randperm(self.num_cells)[:num_cells_to_swap]

        # Perform crossover
        gene_swaps = x[cell_swap_indices][:, gene_swap_indices].clone()
        gene_swaps_shuffled = gene_swaps[torch.randperm(num_cells_to_swap)]
        x[cell_swap_indices][:, gene_swap_indices] = gene_swaps_shuffled
        x.to(device=device, dtype=dtype)
        
        return x

    def instance_crossover_nn(self, x: torch.Tensor, gene_swap_prop: float = 0.2, cell_swap_prop: float = 0.2) -> torch.Tensor:
        """
        Performs a crossover operation between pairs of cells in the tensor.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            gene_swap_prop (float): The proportion of genes to be crossed over.
            cell_swap_prop (float): The proportion of cells to be affected by crossover.

        Returns:
            torch.Tensor: The tensor with crossed-over gene expressions.
        """
       
        dtype, device = x.dtype, x.device
        num_genes_to_swap = int(self.num_genes * gene_swap_prop)
        num_cells_to_swap = int(self.num_cells * cell_swap_prop)
        gene_swap_indices = torch.randperm(self.num_genes)[:num_genes_to_swap]
        cell_swap_indices = torch.randperm(self.num_cells)[:num_cells_to_swap]
        
        # Find the nearest neighbor
        X = x.detach().cpu().numpy()  # Convert tensor to numpy array
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)
        nearest_neighbor_indices = indices[cell_swap_indices, 1]

        # Swap gene expressions between selected cells and their nearest neighbors
        tmp = x[cell_swap_indices][:, gene_swap_indices].clone()
        x[cell_swap_indices][:, gene_swap_indices] = x[nearest_neighbor_indices][:, gene_swap_indices]
        x[nearest_neighbor_indices][:, gene_swap_indices] = tmp
        x = x.to(device=device, dtype=dtype)

        return x

    def bbknn_instance_crossover(self, x, index, connectivities, counts, gene_swap_prop: float = 0.2, cell_swap_prop: float = 0.2) -> torch.Tensor:
        """
        Performs a crossover operation between pairs of cells in the input batch, such that each affected cell is crossed over with one of its neighbors in the BBKNN graph computed on the entire adata object.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            index (np.ndarray) : The original indices (from the adata object) of each cell in the batch x.
            connectivities (np.ndarray): The matrix of pairwise BBKNN connectivities per cell.
            counts (torch.Tensor): tensor of raw read counts for the full original adata object.
            gene_swap_prop (float): The proportion of genes to be crossed over.
            cell_swap_prop (float): The proportion of cells to be affected by crossover.

        Returns:
            torch.Tensor: The tensor with crossed-over gene expressions.
        """
        
      
        dtype, device = x.dtype, x.device
        num_genes_to_swap = int(self.num_genes * gene_swap_prop)
        num_cells_to_swap = int(self.num_cells * cell_swap_prop)
        gene_swap_indices = torch.randint(0, self.num_genes, (num_genes_to_swap,), dtype=torch.long, device=device)
        cell_swap_indices = torch.randint(0, self.num_cells, (num_cells_to_swap,), dtype=torch.long, device=device)

        index = index.to(torch.long)
        
        # Use cont_covs to obtain original indices of selected cells
        original_indices = index[cell_swap_indices]

        rows = connectivities[original_indices]
        
        nonzero_indices = (rows != 0).nonzero()

        # Group the indices by row
        nonzero_indices_grouped = torch.split(nonzero_indices[:, 1], torch.unique(nonzero_indices[:, 0], return_counts=True)[1].tolist())

        # Randomly select one index from each row
        random_nonzero_indices = [torch.randint(0, len(indices), (1,))[0] for indices in nonzero_indices_grouped]

        # Extract the selected indices from the grouped indices
        random_neighbor_indices = torch.stack([indices[idx] for indices, idx in zip(nonzero_indices_grouped, random_nonzero_indices)])

        counts = counts.to(dtype=x.dtype)
        
        # Copy gene counts from neighbors to target cells
        x[cell_swap_indices[:, None], gene_swap_indices] = counts[random_neighbor_indices[:, None], gene_swap_indices]

        return x

    

    def multinomial_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs multinomial augmentation on the tensor by resampling gene expressions.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.

        Returns:
            torch.Tensor: The tensor with resampled read counts.
        """
        dtype, device = x.dtype, x.device
        library_sizes = x.sum(axis=1, dtype=torch.float)
        library_sizes = library_sizes.detach().cpu().numpy().astype(np.float64)

        probs = F.normalize(x.to(torch.float64), p=1, dim=1)
        probs = probs.detach().cpu().numpy().astype(np.float64)
        

        for i in range(self.num_cells):
            random_library_size = np.random.choice(library_sizes, replace=True)
            new_cell = np.random.multinomial(n=random_library_size, pvals=probs[i], size=1)
            x[i] = torch.from_numpy(new_cell)

        x = x.to(dtype=dtype, device=device)
        return x
    

    def multinomial_subsampling(self, x: torch.Tensor, batch_index: torch.Tensor, upsampling: bool = True) -> torch.Tensor:
      """
      Performs multinomial subsampling augmentation on the tensor x based on categorical covariates.

      Args:
          x (torch.Tensor): The tensor containing the scRNA-seq data.
          batch_index (torch.Tensor): The tensor containing batch indices.
          upsampling (bool, optional): If True, perform upsampling for 10X cells, else perform downsampling for SS2 cells.

      Returns:
          torch.Tensor: The tensor with resampled gene expressions.
      """
      tenx_batch_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
      dtype, device = x.dtype, x.device

      # Count the occurrences of each type of batch index
      tenx_mask = torch.tensor([idx in tenx_batch_indices for idx in batch_index])
      tenx_count = torch.sum(tenx_mask)
      ss2_count = batch_index.shape[0] - tenx_count

      # Check if both types of cells are present
      if tenx_count == 0 or ss2_count == 0:
        print("Only one type of cell present. Returning x without subsampling.")
        return x

      # Extract library sizes for 10X and SS2 cells
      tenx_library_sizes = x[tenx_mask].sum(axis=1, dtype=torch.float)
      ss2_library_sizes = x[~tenx_mask].sum(axis=1, dtype=torch.float)

      # Select library sizes based on upsampling flag
      library_sizes = tenx_library_sizes if upsampling else ss2_library_sizes
      library_sizes = library_sizes.detach().cpu().numpy().astype(np.float64)

      # Normalize probabilities
      probs = F.normalize(x.to(torch.float64), p=1, dim=1)
      probs = probs.detach().cpu().numpy().astype(np.float64)

      # Perform subsampling
      for cell_idx in torch.nonzero(tenx_mask if upsampling else ~tenx_mask):
          random_library_size = np.random.choice(library_sizes, replace=True)
          new_cell = np.random.multinomial(n=random_library_size, pvals=probs[cell_idx], size=1)
          x[cell_idx] = torch.from_numpy(new_cell).to(dtype=dtype, device=device)

      return x
    
    def poisson_subsampling(self, x: torch.Tensor, batch_index: torch.Tensor, upsampling: bool = True) -> torch.Tensor:
        """
        Performs Poisson subsampling augmentation on the tensor x based on categorical covariates.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            batch_index (torch.Tensor): The tensor containing batch indices.
            upsampling (bool, optional): If True, perform upsampling for 10X cells, else perform downsampling for SS2 cells.

        Returns:
            torch.Tensor: The tensor with resampled gene expressions.
        """
        tenx_batch_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        dtype, device = x.dtype, x.device

        # Create masks for 10X cells and SS2 cells
        tenx_mask = torch.tensor([idx in tenx_batch_indices for idx in batch_index])
        tenx_count = torch.sum(tenx_mask)
        ss2_count = batch_index.shape[0] - tenx_count
        ss2_mask = ~tenx_mask

        # Check if both types of cells are present
        if tenx_count == 0 or ss2_count == 0:
            print("Only one type of cell present. Returning x without subsampling.")
            return x

        # Compute library sizes for each type of cell
        if upsampling:
            # When upsampling, only resample 10X cells
            library_sizes = x[ss2_mask].sum(axis=1, dtype=torch.float)
            cells_to_resample = tenx_mask 
        else:
            # When downsampling, only resample SS2 cells
            library_sizes = x[tenx_mask].sum(axis=1, dtype=torch.float)
            cells_to_resample = ss2_mask
            
        random_opposite_type_lib_sizes = library_sizes[torch.randint(len(library_sizes), size=(x[cells_to_resample].shape[0],), device=device)]
        ratios = random_opposite_type_lib_sizes / x[cells_to_resample].sum(axis=1) 
        # Resample read counts for relevant cells
        new_read_counts = x.clone()
        new_read_counts[cells_to_resample] = torch.poisson(ratios.view(-1, 1) * x[cells_to_resample])

        return new_read_counts.to(dtype=dtype, device=device)

    def mixup(self, x: torch.Tensor, mixup_percentage: float = 0.2) -> torch.Tensor:
        """
        Performs mixup augmentation on the tensor by randomly replacing old cells with new cells that are a convex combination of 2 randomly selected cells.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            mixup_percentage (float): The percentage of new cells in the tensor to be generated as new cells.

        Returns:
            torch.Tensor: The tensor with added mixup augmented cells.
        """
        
        dtype, device = x.dtype, x.device
        num_new_cells = int(self.num_cells * mixup_percentage)
        indices_to_replace = torch.randperm(self.num_cells)[:num_new_cells]

        X_1 = x
        X_2 = x[torch.randperm(self.num_cells)]

        # Sample lambda parameters from Beta distribution
        lambdas = torch.distributions.Beta(0.2, 0.2).sample((num_new_cells, 1)).to(device)

        # Generate new cells as convex combination of X_1 and X_2
        new_cells = lambdas * X_1[indices_to_replace] + (1 - lambdas) * X_2[indices_to_replace]

        # Overwrite randomly selected cells with new cells
        x[indices_to_replace] = new_cells.to(dtype=dtype, device=device)

        return x


    def random_power_transformation(self, x: torch.Tensor, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Applies a random power function augmentation to the tensor with a certain probability.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            augmentation_probability (float): Probability of applying the augmentation.

        Returns:
            torch.Tensor: The tensor with randomly applied power function augmentation.
        """
        
        dtype, device = x.dtype, x.device
        a = 0.8 + (2 - 0.8) * torch.rand(1)
        b = 1.0 + (2 - 1.0) * torch.rand(1)
        a = a.to(device)
        b = b.to(device)
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()

        # Apply power function transformation to the tensor
        x[augmented_cell_idxs] = a * torch.pow(x[augmented_cell_idxs], b)
        x = x.to(device=device, dtype=dtype)

        return x
    
    def random_linear_transformation(self, x: torch.Tensor, apply_prob: float = 0.5) -> torch.Tensor:
        """
        Applies a random linear function augmentation to the tensor with a certain probability.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq data.
            apply_prob (float): Probability of applying the augmentation.

        Returns:
            torch.Tensor: The tensor with randomly applied linear function augmentation.
        """

        dtype, device = x.dtype, x.device
        a = 0.5 + (3 - 0.5) * torch.rand(1)
        a = a.to(device)
        augmented_cell_idxs = (torch.rand(self.num_cells) < apply_prob).nonzero().squeeze()

        # Apply linear function transformation to the tensor
        x[augmented_cell_idxs] = a * x[augmented_cell_idxs]
        x = x.to(device=device, dtype=dtype)

        return x
    
    def apply_augmentations(self, x=None, augmentation_to_apply=None, batch_index=None, mask_probs=None, cont_covs=None, connectivities=None, counts=None):
        """
        Applies a chosen augmentation to a batch x.

        Args:
            x (torch.Tensor): The tensor containing the scRNA-seq training batch.
            augmentation_to_apply (List[str]): A list containing a string that specifies which augmentation(s) to apply to x.
            batch_index (torch.Tensor): The index of every cell in the current batch.
            mask_probs (np.ndarray): The densities (a.k.a 1 - sparsity) of each gene across the entire dataset. 
            cont_covs (np.ndarray) : The original indices (from the adata object) of each cell in the batch x.
            connectivities (np.ndarray): The matrix of pairwise BBKNN connectivities per cell.
            counts (torch.Tensor): tensor of raw read counts for the full original adata object..

        Returns:
            torch.Tensor: The tensor with randomly applied linear function augmentation.
        """
        
        # Define a dictionary mapping augmentation names to methods
        augmentation_methods = {
            'random_mask': self.random_mask,
            'random_swap': self.random_swap,
            'instance_crossover': self.instance_crossover,
            'instance_crossover_nn': self.instance_crossover_nn,
            'multinomial_sampling': self.multinomial_sampling,
            'mixup': self.mixup,
            'random_power_transformation': self.random_power_transformation,
            'log_normal_noise': self.log_normal_noise,
            'poisson_noise': self.poisson_noise,
            'random_linear_transformation': self.random_linear_transformation,
            'multinomial_subsampling': self.multinomial_subsampling,
            'poisson_subsampling': self.poisson_subsampling,
            'sparsity_based_masking': self.sparsity_based_masking,
            'bbknn_instance_crossover': self.bbknn_instance_crossover
            # Add other augmentation methods here if needed
        }

        used_augmentations = set()

        for augmentation_name in augmentation_to_apply:
            if augmentation_name == "None":
                return x
            if augmentation_name not in augmentation_methods:
                raise ValueError(f"Unknown augmentation: {augmentation_name}")

            # Apply the augmentation
            if augmentation_name == "multinomial_subsampling":
                x = self.multinomial_subsampling(x, batch_index)
            elif augmentation_name == "poisson_subsampling":
                x = self.poisson_subsampling(x, batch_index)
            elif augmentation_name == "sparsity_based_masking":
                x = self.sparsity_based_masking(x, mask_probs)
            elif augmentation_name == "bbknn_instance_crossover":
                x = self.bbknn_instance_crossover(x, cont_covs, connectivities, counts)
            else:
                x = augmentation_methods[augmentation_name](x)
            used_augmentations.add(augmentation_name)

        # print(f"Augmentations Used: {used_augmentations}")
        return x


class VAE(EmbeddingModuleMixin, BaseMinifiedModeModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the scVI model described in :cite:p:`Lopez18`.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    batch_representation
        ``EXPERIMENTAL`` How to encode batch labels in the data. One of the following:

        * ``"one-hot"``: represent batches with one-hot encodings.
        * ``"embedding"``: represent batches with continuously-valued embeddings using :class:`~scvi.nn.Embedding`.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    extra_encoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
    batch_embedding_kwargs
        Keyword arguments passed into :class:`~scvi.nn.Embedding` if ``batch_representation`` is set to ``"embedding"``.

    Notes
    -----
    Lifecycle: argument ``batch_representation`` is experimental in v1.2.
    """

    def __init__(
        self,
        n_input: int,
        augmentation_to_apply: Optional[List[str]] = None,
        adata: AnnData | None = None,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Callable = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        batch_embedding_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.adata = adata

        # Compute gene sparsity once for better efficiency
        if not isinstance(self.adata.layers["counts"], np.ndarray):
          read_counts = self.adata.layers["counts"].toarray()
        else:
          read_counts = self.adata.layers["counts"]
          
        zero_counts = np.sum(read_counts == 0, axis=0)
        sparsity = zero_counts / read_counts.shape[0]
        mask_probs = 1 - sparsity
        self.mask_probs = mask_probs

        # Compute BBKNN graph
        neighbors_within_batch = 25 if adata.n_obs > 100000 else 3
        adata.X = adata.layers["logcounts"].copy()
        sc.pp.pca(adata)
        bbknn.bbknn(
            adata, batch_key=REGISTRY_KEYS.BATCH_KEY, neighbors_within_batch=neighbors_within_batch
        )

        # Extract BBKNN connectivities
        self.connectivities = adata.obsp["connectivities"]


        # Convert our read count matrix to the right format
        counts = adata.layers["counts"]
        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts).to(dtype=torch.int, device=torch.device("cuda"))
        else:
            counts = torch.from_numpy(counts.toarray()).to(dtype=torch.int, device=torch.device("cuda"))
        self.counts = counts

        # Convert our connectivities matrix to the right format
        if not isinstance(self.connectivities, np.ndarray):
            self.connectivities = torch.tensor(self.connectivities.toarray())
            self.connectivities = self.connectivities.to(device=torch.device("cuda"))

        

        self.augmentation_to_apply = augmentation_to_apply
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_continuous_cov = 0
        encode_covariates = False
        n_input_encoder = n_input + n_continuous_cov * encode_covariates

        if self.batch_representation == "embedding":
            # batch embeddings are concatenated to the input of the encoder
            n_input_encoder += batch_dim * encode_covariates
            # don't pass in batch index if using embeddings
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = {} ##extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            # batch embeddings are concatenated to the input of the decoder
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.minified_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            input_dict = {
                "x": x,
                "batch_index": batch_index,
                "cont_covs": cont_covs,
                "cat_covs": cat_covs,
            }
        else:
            if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                observed_lib_size = tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE]
                input_dict = {
                    "qzm": qzm,
                    "qzv": qzv,
                    "observed_lib_size": observed_lib_size,
                }
            else:
                raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
    

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None
        )

        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor,
        }
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
        local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        
        post_log1p_augmentations = ["mixup", "random_power_transformation", "random_linear_transformation", "log_normal_noise"]

        
        
        
        if self.log_variational:
            if self.augmentation_to_apply is not None:
                augmentation = Augmentation(x_)
                if self.augmentation_to_apply[0] in post_log1p_augmentations:
                    x_ = torch.log1p(x_)
                    x_ = augmentation.apply_augmentations(x_, self.augmentation_to_apply, batch_index, self.mask_probs, cont_covs)
                else:
                    x_ = augmentation.apply_augmentations(x_, self.augmentation_to_apply, batch_index, self.mask_probs, cont_covs, self.connectivities, self.counts)
                    x_ = torch.log1p(x_)
            else:
                x_ = torch.log1p(x_)

        #if cont_covs is not None and self.encode_covariates:
         #   encoder_input = torch.cat((x_, cont_covs), dim=-1)
        #else:
        encoder_input = x_

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
    

        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    @auto_move_data
    def _cached_inference(self, qzm, qzv, observed_lib_size, n_samples=1):
        if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            dist = Normal(qzm, qzv.sqrt())
            # use dist.sample() rather than rsample because we aren't optimizing the z here
            untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = torch.log(observed_lib_size)
            if n_samples > 1:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")
        outputs = {"z": z, "qz_m": qzm, "qz_v": qzv, "ql": None, "library": library}
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""

        #if cont_covs is None:
         #   decoder_input = z
        #elif z.dim() != cont_covs.dim():
         #   decoder_input = torch.cat(
          #      [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
           # )
        #else:
         #  decoder_input = torch.cat([z, cont_covs], dim=-1)

        decoder_input = z

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
            )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {
            "px": px,
            "pl": pl,
            "pz": pz,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
        max_poisson_rate: float = 1e8,
    ) -> torch.Tensor:
        r"""Generate predictive samples from the posterior predictive distribution.

        The posterior predictive distribution is denoted as :math:`p(\hat{x} \mid x)`, where
        :math:`x` is the input data and :math:`\hat{x}` is the sampled data.

        We sample from this distribution by first sampling ``n_samples`` times from the posterior
        distribution :math:`q(z \mid x)` for a given observation, and then sampling from the
        likelihood :math:`p(\hat{x} \mid z)` for each of these.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into :meth:`~scvi.module.VAE.forward`.
        n_samples
            Number of Monte Carlo samples to draw from the distribution for each observation.
        max_poisson_rate
            The maximum value to which to clip the ``rate`` parameter of
            :class:`~torch.distributions.Poisson`. Avoids numerical sampling
            issues when the parameter is very large due to the variance of the
            distribution.

        Returns
        -------
        Tensor on CPU with shape ``(n_obs, n_vars)`` if ``n_samples == 1``, else
        ``(n_obs, n_vars,)``.
        """
        inference_kwargs = {"n_samples": n_samples}
        _, generative_outputs = self.forward(
            tensors, inference_kwargs=inference_kwargs, compute_loss=False
        )

        dist = generative_outputs["px"]
        if self.gene_likelihood == "poisson":
            dist = torch.distributions.Poisson(torch.clamp(dist.rate, max=max_poisson_rate))

        # (n_obs, n_vars) if n_samples == 1, else (n_samples, n_obs, n_vars)
        samples = dist.sample()
        # (n_samples, n_obs, n_vars) -> (n_obs, n_vars, n_samples)
        samples = torch.permute(samples, (1, 2, 0)) if n_samples > 1 else samples

        return samples.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(
        self,
        tensors,
        n_mc_samples,
        return_mean=False,
        n_mc_samples_per_pass=1,
    ):
        """Computes the marginal log likelihood of the model.

        Parameters
        ----------
        tensors
            Dict of input tensors, typically corresponding to the items of the data loader.
        n_mc_samples
            Number of Monte Carlo samples to use for the estimation of the marginal log likelihood.
        return_mean
            Whether to return the mean of marginal likelihoods over cells.
        n_mc_samples_per_pass
            Number of Monte Carlo samples to use per pass. This is useful to avoid memory issues.
        """
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            logger.warn(
                "Number of chunks is larger than the total number of samples, setting it to the number of samples"
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(
                tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass}
            )
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl


class LDVAE(VAE):
    """Linear-decoded Variational auto-encoder model.

    Implementation of :cite:p:`Svensson20`.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset. It may also be used
    for all scVI tasks, like differential expression, batch correction, imputation, etc.
    However, batch correction may be less powerful as it assumes a linear model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution.
    **kwargs
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        use_observed_lib_size: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            **kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
        )

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings
