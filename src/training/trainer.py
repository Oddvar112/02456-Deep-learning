import os
import torch
import torch.nn as nn
from src.training.loss import DPRLoss

class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, train_loader, val_loader, epochs: int = 10, lr: float = 1e-4, device: str = 'cpu', weights_dir: str = 'outputs/weights'):
        
        # Move the models to CPU or GPU
        query_encoder.to(device)
        document_encoder.to(device)
        
        query_optimizer = torch.optim.AdamW(query_encoder.parameters(), lr=lr)
        document_optimizer = torch.optim.AdamW(document_encoder.parameters(), lr=lr)

        history = {'train_loss': [], 'val_loss': []}  
        best_val_loss = float('inf')                  

        for epoch in range(epochs):  

            query_encoder.train()     # training mode
            document_encoder.train()

            total_loss = 0.0
            for queries, documents in train_loader:
                queries, documents = queries.to(device), documents.to(device)  # data must be on the same device as the models

                loss = self.criterion(query_encoder(queries), document_encoder(documents))

                query_optimizer.zero_grad()     
                document_optimizer.zero_grad()
                loss.backward()                 
                query_optimizer.step()          
                document_optimizer.step()

                total_loss += loss.item()  # .item() turns the loss into a normal number not a tensor
            history['train_loss'].append(total_loss / len(train_loader))

            # Validation: how good the encoders from this epoch are on data they have not trained on
            history['val_loss'].append(self.validate(query_encoder, document_encoder, val_loader, device))

            print(f'Epoch {epoch + 1}/{epochs}  train loss: {history["train_loss"][-1]:.4f}  val loss: {history["val_loss"][-1]:.4f}')

            # Save the weights when the validation loss is the best so far
            if history['val_loss'][-1] < best_val_loss:
                best_val_loss = history['val_loss'][-1]
                self.save_weights(query_encoder, document_encoder, weights_dir)

        return history

    def validate(self, query_encoder: nn.Module, document_encoder: nn.Module, val_loader, device: str):
        """
        Returns the average loss on val_loader, using the encoders as they are after the latest training epoch.
        """
        query_encoder.eval()      # evaluation mode 
        document_encoder.eval()
        total_loss = 0.0
        with torch.no_grad():    
            for queries, documents in val_loader:
                queries, documents = queries.to(device), documents.to(device)
                total_loss += self.criterion(query_encoder(queries), document_encoder(documents)).item()
        return total_loss / len(val_loader)

    def save_weights(self, query_encoder: nn.Module, document_encoder: nn.Module, weights_dir: str):
        """
        Saves the weights of both encoders
        """
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(query_encoder.state_dict(), os.path.join(weights_dir, 'query_encoder.pt'))
        torch.save(document_encoder.state_dict(), os.path.join(weights_dir, 'document_encoder.pt'))
