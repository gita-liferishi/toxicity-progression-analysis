from tqdm import tqdm
from transformer import logging

# Define lists to store logs
training_logs = {
    "epoch": [],
    "train_loss": [],
    "val_loss": []
}

num_epochs = 5
best_val_loss = float('inf')
patience = 3

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        weights = batch['weights'].to(device)
        comment_nature_features = batch['comment_nature_features'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, comment_nature_features=comment_nature_features)
        loss = criterion(outputs.squeeze(-1), labels)
        weighted_loss = (loss * weights).mean()

        # Backward pass
        weighted_loss.backward()
        optimizer.step()

        total_train_loss += weighted_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device)
            comment_nature_features = batch['comment_nature_features'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, comment_nature_features=comment_nature_features)
            loss = criterion(outputs.squeeze(-1), labels)

            # Compute loss
            weighted_loss = (loss * weights).mean()
            val_loss += weighted_loss.item()

    avg_val_loss = val_loss / len(val_loader)        
    # Store metrics
    training_logs["epoch"].append(epoch + 1)
    training_logs["train_loss"].append(avg_train_loss)
    training_logs["val_loss"].append(avg_val_loss)
    
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
        print(f"Validation loss improved. Saving model at epoch {epoch + 1}.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

# Load the best model state before exiting
model.load_state_dict(best_model_state)
print("Training stopped. Best model loaded.")