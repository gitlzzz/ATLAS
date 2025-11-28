"""Train an autoencoder for dimensionality reduction."""

import pathlib as pl

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import MatDBForge.active_learning.extrapolation.autoencoder as ae
import MatDBForge.core.code_utils as mdb_cut


def train_loop(data_loader, model, loss_fn, optimizer):
    num_batches = len(data_loader)
    # Set the model to training mode - important for batch normalization
    # and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    running_loss = 0.0

    for _, X in enumerate(data_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # model(X): Forward pass, compute the reconstruction
        # Compute loss: reconstruction error
        loss = loss_fn(model(X), X)

        # Backward pass: compute gradients
        loss.backward()

        # Optimize: update model weights
        optimizer.step()

        # Track loss
        running_loss += loss.item()

    # Compute average loss
    avg_loss = running_loss / num_batches

    return running_loss, avg_loss


def val_loop(data_loader, model, loss_fn):
    # Set the model to evaluation mode - important for batch
    # normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(data_loader)
    running_loss = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients
    # are computed during test mode also serves to reduce unnecessary gradient
    # computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X in data_loader:
            # Forward pass: compute the reconstruction
            reconstruction = model(X)

            # Compute loss: reconstruction error
            loss = loss_fn(reconstruction, X)

            # Track loss
            running_loss += loss.item()

    # Compute average loss
    avg_loss = running_loss / num_batches

    return avg_loss


def load_dataset(
    data: str | np.ndarray,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    rng_seed: int = None,
    device: str = None,
    dtype=torch.float32,
):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the RNG seed for reproducibility
    if not rng_seed:
        rng_seed = np.random.randint(1, int(1e15))

    # Load the dataset
    if isinstance(data, (str, pl.Path)):
        point_arr = np.load(data)
    elif isinstance(data, np.ndarray):
        point_arr = data

    # If npz file, extract the first array
    # This is a workaround for the npz file format
    # that stores the data in a dictionary-like structure
    # with the key 'arr_0'
    if isinstance(point_arr, np.lib.npyio.NpzFile):
        point_arr = point_arr.get('descriptor')

        # Old name fallback
        if point_arr is None:
            point_arr = point_arr.get('arr_0')

        if point_arr is None:
            raise ValueError("No array found in the npz file.")

    valid_data_size = int(point_arr.shape[0] * valid_frac)
    test_data_size = int(point_arr.shape[0] * test_frac)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=rng_seed)
    torch.manual_seed(rng_seed)

    # Select a fraction of the data for training, validation, and testing
    valid_arr_idx = rng.choice(
        point_arr.shape[0],
        size=valid_data_size,
        replace=False,
    )

    # Remove validation data from the point array
    valid_arr = point_arr[valid_arr_idx]

    # Remove selected values from the point_arr
    point_arr = np.delete(point_arr, obj=valid_arr_idx, axis=0)

    # Getting the test data
    test_arr_idx = rng.choice(
        point_arr.shape[0],
        size=test_data_size,
        replace=False,
    )

    test_arr = point_arr[test_arr_idx]
    point_arr = np.delete(point_arr, obj=test_arr_idx, axis=0)

    train_data = torch.Tensor(point_arr).to(device=device, dtype=dtype)
    valid_data = torch.Tensor(valid_arr).to(device=device, dtype=dtype)
    test_data = torch.Tensor(test_arr).to(device=device, dtype=dtype)
    mdb_cut.custom_print('Data loaded:', 'info')
    mdb_cut.custom_print(f'  Training data array shape: {train_data.shape}', 'clean')
    mdb_cut.custom_print(f'  Validation data array shape: {valid_data.shape}', 'clean')
    mdb_cut.custom_print(f'  Test data array shape: {test_data.shape}', 'clean')
    print()

    return train_data, valid_data, test_data


def run_training(args):
    """
    Train an autoencoder model for dimensionality reduction.

    Parameters
    ----------
    args : Namespace
        Namespace object containing the following attributes:
        - dataset : str
            Path to the dataset file.
        - device : str
            Device to run the model on.
            If not given, the device is set to cuda if available, else cpu.
        - dtype : str
            Data type to use for the model. One of float32 or float64.
            Default is float32.
        - lr : float
            Learning rate for the optimizer.
        - num_epochs : int
            Number of epochs to train the model.
        - batch_size : int
            Batch size for training.
        - patience : int
            Number of epochs to wait before reducing the learning rate.
        - train_frac : float
            Fraction of the dataset to use for training.
        - valid_frac : float
            Fraction of the dataset to use for validation.
        - test_frac : float
            Fraction of the dataset to use for testing.
        - l1_hidden_dim : int
            Number of hidden units in the first layer of the autoencoder.
        - l2_hidden_dim : int
            Number of hidden units in the second layer of the autoencoder.
        - weight_decay : float
            Weight decay for the optimizer.
        - bias_flag : bool
            Flag to include bias in the linear layers.
        - verbose : bool
            Flag to print verbose output.
        - rng_seed : int
            Random seed for reproducibility.
        - model_path : str
            Path to save the trained model.
        - wandb : bool
            Flag to enable logging to wandb.
        - wandb_project : str
            Name of the wandb project.
        - wandb_name : str
            Name of the wandb run.


    Returns
    -------
    Autoencoder
        Autoencoder model trained for dimensionality reduction.
    """
    # Set device if no device is given
    if not hasattr(args, 'device') or args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    mdb_cut.custom_print(f"Running on device: '{device}'", 'info')

    # Setting verbosity
    if 'verbose' in vars(args):
        match args.verbose:
            case True:
                args.verbose = True
            case False:
                args.verbose = False
    else:
        args.verbose = True

    # Setting dtype
    if not hasattr(args, 'dtype'):
        args.dtype = torch.float32
    else:
        match args.dtype:
            case 'float32':
                args.dtype = torch.float32
            case 'float64':
                args.dtype = torch.float64
            case _:
                args.dtype = torch.float32
    mdb_cut.custom_print(f"Using dtype: '{args.dtype}'", 'info')

    # If no seed is given, generate a random seed
    if not hasattr(args, 'rng_seed'):
        args.rng_seed = np.random.randint(1, int(1e15))
    mdb_cut.custom_print(f"Using RNG seed: '{args.rng_seed}'.", 'info')

    # Load data
    train_data, valid_data, test_data = load_dataset(
        data=args.dataset,
        device=device,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        rng_seed=args.rng_seed,
        dtype=args.dtype,
    )

    # Number of input dimensions
    input_dim = train_data.shape[1]

    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # set the name of the run
            name=args.wandb_name,
            # track hyperparameters and run metadata
            config={
                'learning_rate': args.lr,
                'bias_flag': args.bias_flag,
                'dataset': args.dataset,
                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'patience': args.patience,
                'train_frac': args.train_frac,
                'valid_frac': args.valid_frac,
                'test_frac': args.test_frac,
                'l1_hidden_dim': args.l1_hidden_dim,
                'l2_hidden_dim': args.l2_hidden_dim,
                'weight_decay': args.weight_decay,
                'device': device,
            },
        )

    # Model initialization
    model = ae.Autoencoder(
        input_dim=input_dim,
        l1_dim=args.l1_hidden_dim,
        l2_dim=args.l2_hidden_dim,
        bias_flag=args.bias_flag,
    )
    model.to(device=device, dtype=args.dtype)

    match args.loss:
        # Define loss function as MSE (Mean Squared Error)
        case 'mse':
            criterion = nn.MSELoss()

        # Alternative loss function that penalizes small errors more heavily
        case 'weighted_mse':

            def weighted_mse_loss(output, target, weight=10):
                diff = (output - target) ** 2
                return torch.mean(weight * diff)

            criterion = weighted_mse_loss

        # Default is MSE
        case _:
            criterion = nn.MSELoss()

    # Define optimizer (Adam optimizer)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience,
        # threshold=1e-3,
    )

    # Create a DataLoader for batch processing
    train_data = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    val_data = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        # shuffle=True,
    )

    mdb_cut.custom_print(
        f'Starting autoencoder training for {args.num_epochs} epochs...', 'info'
    )

    for epoch in range(args.num_epochs):
        _, train_avg_loss = train_loop(
            data_loader=train_data,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
        )

        val_loss = val_loop(
            data_loader=val_data,
            model=model,
            loss_fn=criterion,
        )

        # Perform test evaluation every 5 epochs
        # Specifically
        if epoch % 5 == 0:
            test_loss = val_loop(
                data_loader=test_data,
                model=model,
                loss_fn=criterion,
            )
            if args.verbose:
                mdb_cut.custom_print(
                    {
                        'Epoch': epoch,
                        'Train Avg. MSE': train_avg_loss,
                        'Validation Avg. MSE': val_loss,
                        'Test Avg. MSE': test_loss,
                        'lr': scheduler.get_last_lr()[0],
                    },
                    'none',
                )
            # log metrics to wandb
            if args.wandb:
                wandb.log(
                    {
                        'epoch': epoch,
                        'train_loss': train_avg_loss,
                        'val_loss': val_loss,
                        'test_loss': val_loss,
                        'lr': scheduler.get_last_lr()[0],
                    }
                )
        # Log metrics for normal epoch
        else:
            if args.verbose:
                mdb_cut.custom_print(
                    {
                        'Epoch': epoch,
                        'Train Avg. MSE': train_avg_loss,
                        'Validation Avg. MSE': val_loss,
                        'lr': scheduler.get_last_lr()[0],
                    },
                    'none',
                )
            # log metrics to wandb
            if args.wandb:
                wandb.log(
                    {
                        'epoch': epoch,
                        'train_loss': train_avg_loss,
                        'val_loss': val_loss,
                        'lr': scheduler.get_last_lr()[0],
                    }
                )

        scheduler.step(metrics=val_loss)

    mdb_cut.custom_print('Training complete!', 'done')

    if hasattr(args, 'model_path'):
        # Save the model
        # torch.save(model.state_dict(), args.model_path)
        save_path = pl.Path(args.model_path).absolute()
        torch.save(model, save_path)

        mdb_cut.custom_print(f"Autoencoder model saved to '{save_path}'.", 'info')
    else:
        mdb_cut.custom_print('Autoencoder model not saved.', 'warning')

    return model
