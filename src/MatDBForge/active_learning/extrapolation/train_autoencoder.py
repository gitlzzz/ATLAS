"""Train an autoencoder for dimensionality reduction."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import MatDBForge.active_learning.extrapolation.autoencoder as ae
import MatDBForge.core.code_utils as mdb_cud


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
    data_path,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    device,
    rng_seed: int,
):
    point_arr = np.load(data_path)

    valid_data_size = int(point_arr.shape[0] * valid_frac)
    test_data_size = int(point_arr.shape[0] * test_frac)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=rng_seed)
    torch.manual_seed(rng_seed)

    # Select a fraction of the data for training, validation, and testing
    valid_arr_idx = rng.random.choice(
        point_arr.shape[0],
        size=valid_data_size,
        replace=False,
    )

    # Remove validation data from the point array
    valid_arr = point_arr[valid_arr_idx]

    # Remove selected values from the point_arr
    point_arr = np.delete(point_arr, obj=valid_arr_idx, axis=0)

    # Getting the test data
    test_arr_idx = rng.random.choice(
        point_arr.shape[0],
        size=test_data_size,
        replace=False,
    )

    test_arr = point_arr[test_arr_idx]
    point_arr = np.delete(point_arr, obj=test_arr_idx, axis=0)

    train_data = torch.Tensor(point_arr).to(device=device)
    valid_data = torch.Tensor(valid_arr).to(device=device)
    test_data = torch.Tensor(test_arr).to(device=device)
    mdb_cud.custom_print("Data loaded:", "info")
    mdb_cud.custom_print(f"- Training data array shape: {train_data.shape}", "clean")
    mdb_cud.custom_print(f"- Validation data array shape: {valid_data.shape}", "clean")
    mdb_cud.custom_print(f"- Test data array shape: {test_data.shape}", "clean")
    print()

    return train_data, valid_data, test_data


def run_training(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdb_cud.custom_print(f"Running on device: {device}", "info")

    # Load data
    train_data, valid_data, test_data = load_dataset(
        data_path=args.dataset,
        device=device,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        rng_seed=args.seed,
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
                "learning_rate": args.lr,
                "bias_flag": args.bias_flag,
                "dataset": args.dataset,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "train_frac": args.train_frac,
                "valid_frac": args.valid_frac,
                "test_frac": args.test_frac,
                "l1_hidden_dim": args.l1_hidden_dim,
                "l2_hidden_dim": args.l2_hidden_dim,
                "weight_decay": args.weight_decay,
                "device": device,
            },
        )

    # Model initialization
    model = ae.Autoencoder(
        input_dim=input_dim,
        l1_dim=args.l1_hidden_dim,
        l2_dim=args.l2_hidden_dim,
        bias_flag=args.bias_flag,
    )
    model.to(device=device)

    match args.loss:
        # Define loss function as MSE (Mean Squared Error)
        case "mse":
            criterion = nn.MSELoss()

        # Alternative loss function that penalizes small errors more heavily
        case "weighted_mse":

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
        mode="min",
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

    mdb_cud.custom_print(f"Starting training for {args.num_epochs} epochs...", "info")

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
            mdb_cud.custom_print(
                {
                    "Epoch": epoch,
                    "Train Avg. MSE": train_avg_loss,
                    "Validation Avg. MSE": val_loss,
                    "Test Avg. MSE": test_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                "info",
            )
            # log metrics to wandb
            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_avg_loss,
                        "val_loss": val_loss,
                        "test_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
        # Log metrics for normal epoch
        else:
            mdb_cud.custom_print(
                {
                    "Epoch": epoch,
                    "Train Avg. MSE": train_avg_loss,
                    "Validation Avg. MSE": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                "info",
            )
            # log metrics to wandb
            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_avg_loss,
                        "val_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

        scheduler.step(metrics=val_loss)

    # Save the model
    torch.save(model.state_dict(), args.model_path)

    mdb_cud.custom_print(
        f"Training complete! Model saved to '{args.model_path}'.", "done"
    )
