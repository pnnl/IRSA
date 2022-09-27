import torch


def train(model, train_loader, val_loader, num_epochs, criterion):
    train_losses = []
    val_losses = []
    cur_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        print("Starting epoch " + str(epoch+1))
        for spec1, spec2, labels in train_loader:

            # Forward
            spec1 = spec1.to(device)
            spec2 = spec2.to(device)
            labels = labels.to(device)
            outputs = model(spec1, spec2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_running_loss = 0.0

        # check validation loss after every epoch
        with torch.no_grad():
            model.eval()

            for spec1, spec2, labels in val_loader:
                spec1 = spec1.to(device)
                spec2 = spec2.to(device)
                labels = labels.to(device)
                outputs = model(spec1, spec2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        pstring = 'Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.8f}'
        print(pstring.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))

    print("Finished Training")
    return train_losses, val_losses
 