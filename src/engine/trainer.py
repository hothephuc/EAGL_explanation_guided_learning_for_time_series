import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
import json

def train_model(cfg, model, train_loader, val_loader):
    device = cfg['model']['device']
    learning_rate = cfg['train']['lr']
    num_epochs = cfg['train']['epochs']
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_labels.append(labels.cpu())
                all_predictions.append(predicted.cpu())
        
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * (all_predictions == all_labels).sum() / all_labels.size

        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg['train']['save_model_path'], 'best_model.pth'))
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        with open(os.path.join(cfg['train']['save_model_path'], 'hist.json'), "w") as outfile: 
            json.dump(history, outfile)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(classification_report(all_labels, all_predictions, digits=4))
        print('-' * 50)
    
    return model, history, best_val_acc