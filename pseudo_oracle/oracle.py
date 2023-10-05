import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class SentenceClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(SentenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class OracleDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    def __len__(self):
        return len(self.data)

def train_oracle(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_oracle(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels.extend(batch['label'])
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.scores, dim=1)
            predictions.extend(predicted)
    correct = 0
    total = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy

def main(data):

    # Tokenize the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 512
    data['text'] = data['text'].apply(lambda x: tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    ))

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create the datasets and dataloaders
    train_dataset = OracleDataset(train_data, tokenizer, max_len)
    val_dataset = OracleDataset(val_data, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, optimizer, and scheduler
    model = SentenceClassifier(768, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Train the model
    for epoch in range(5):
        train_loss = train_oracle(model, train_dataloader, optimizer, device)
        val_accuracy = evaluate_oracle(model, val_dataloader, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        scheduler.step(val_accuracy)

    # Save the model
    torch.save(model.state_dict(), 'sentence_classifier.pth')

if __name__ == '__main__':
    main()