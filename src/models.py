import torch
import torch.nn as nn
from transformers import AutoModel

# Choose Base Model
# model_name = 'bert-base-uncased'
# model_name = 'roberta-base'
# model_name = 'distilbert-base-uncased'
# model_name = 'microsoft/deberta-base'
# model_name = 'xlnet-base-cased'

# ====================================
# Custom Model for Toxicity Regression
# ====================================
bilstm = False
class CustomToxicityModel(nn.Module):
    def __init__(self, model_name, num_comment_nature_features):
        super(CustomToxicityModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(self.transformer.config.hidden_size + num_comment_nature_features, 1)

    def forward(self, input_ids, attention_mask, comment_nature_features):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_output.last_hidden_state[:, 0, :]  # CLS token
        cls_embedding = self.dropout(cls_embedding)
        combined = torch.cat((cls_embedding, comment_nature_features), dim=1)
        output = self.fc(combined)
        return output

# =============================
# Base Model with Bi-LSTM Layer
# =============================
bilstm = True
class BERT_BiLSTM(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_labels, num_comment_nature_features):
        super(BERT_BiLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.25)
        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=768,  # BERT hidden size
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        # Fully connected layer: concatenates BiLSTM output and additional features
        self.fc = nn.Linear(hidden_dim * 2 + num_comment_nature_features, num_labels)  # Concatenate

    def forward(self, input_ids, attention_mask, comment_nature_features):
        # BERT forward pass
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # BiLSTM forward pass
        lstm_output, (hn, cn) = self.lstm(sequence_output)
        lstm_output = lstm_output[:, -1, :]  # Take the last token's output

        # Concatenate comment nature features
        concatenated_features = torch.cat((lstm_output, comment_nature_features), dim=1)

        # Fully connected layer
        logits = self.fc(concatenated_features)

        return logits

# Initialize model
num_comment_nature_features = len(comment_nature_columns)
if bilstm:
  hidden_dim = 128  # Size of LSTM hidden layer
  model = BERT_BiLSTM(bert_model=model_name, hidden_dim=hidden_dim, num_labels=1,  num_comment_nature_features=num_comment_nature_features)
else:
  model = CustomToxicityModel(model_name, num_comment_nature_features)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Move model to current device
model.to(device)
