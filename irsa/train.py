from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn


def train(model, train_loader, val_loader, num_epochs):
    # Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize network
    print(model)

    # Send to device
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1E-3)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Control initialization
    best_val_loss = float("Inf")

    # Loss history containers
    train_losses = []
    val_losses = []

    # Iterate epochs
    for epoch in range(num_epochs):
        # Model in train mode
        model.train()

        # Training accumulators
        running_train_loss = 0.0
        running_train_acc = 0.0

        # Iterate over batches
        for batch in train_loader:
            # Send to device
            batch = OrderedDict([(key, value.to(device))
                                for key, value in batch.items() if '_label' not in key])

            # Predict
            outputs = model(batch['exp_spec'], batch['pred_spec'])

            # Calculate loss
            loss = loss_fn(outputs, batch['label'])
            running_train_loss += loss.item()

            # Calculate accuracy
            acc = (batch['label'] == (outputs > 0.5)
                   ).sum().item() / len(outputs)
            running_train_acc += acc

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        with torch.no_grad():
            # Model in evaluation mode
            model.eval()

            # Validation accumulators
            running_val_loss = 0.0
            running_val_acc = 0.0

            # Iterate validation instances
            for batch in val_loader:
                # Send to device
                batch = OrderedDict(
                    [(key, value.to(device)) for key, value in batch.items() if '_label' not in key])

                # Predict
                outputs = model(batch['exp_spec'], batch['pred_spec'])

                # Calculate loss
                loss = loss_fn(outputs, batch['label'])
                running_val_loss += loss.item()

                # Calculate accuracy
                acc = (batch['label'] == (outputs > 0.5)).sum().item()
                running_val_acc += acc / len(outputs)

        # Average training metrics
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_acc = running_train_acc / len(train_loader)

        # Keep training loss
        train_losses.append(avg_train_loss)

        # Average validation metrics
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_acc = running_val_acc / len(val_loader)

        # Keep track of validation loss
        val_losses.append(avg_val_loss)

        # Save best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model/ir_exp_pred_pnn.pt')
            print('**Epoch [{}/{}] Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}**'
                  .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))
        else:
            print('Epoch [{}/{}] Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))

    print("Finished Training")
    return train_losses, val_losses
