import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            # Move data to GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device)
            comment_nature_features = batch['comment_nature_features'].to(device)

            # Make predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, comment_nature_features=comment_nature_features)
            predictions = outputs.squeeze(-1).cpu().numpy()

            # Collect predictions and labels
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    mse = mean_squared_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)

    return mae, mse, r2, all_predictions, all_labels

# Test evaluation
mae, mse, r2, predictions, labels = evaluate_model(model, test_loader, device)

print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")