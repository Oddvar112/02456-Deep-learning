import os
import json
import torch

from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
from src.data_utils.dataloader import tokenize


class DPR(Retriever):
    def __init__(self, cfg, corpus=None):
        super().__init__(corpus.all_documents)
        self.cfg = cfg
        
        # Read weight paths from config (no hardcoding)
        weights_dir = getattr(cfg.retriever, 'weights_dir', 'outputs/weights')
        q_file  = getattr(cfg.retriever, 'query_encoder_weights_file', 'query_encoder.pt')
        d_file  = getattr(cfg.retriever, 'document_encoder_weights_file', 'document_encoder.pt')
        
        self.query_weights_path = os.path.join(weights_dir, q_file)
        self.document_weights_path = os.path.join(weights_dir, d_file)
        
        # Check if trained weights exist; auto-train if missing and corpus is available
        weights_exist = (os.path.exists(self.query_weights_path) and 
                         os.path.exists(self.document_weights_path))
        
        if not weights_exist and corpus is not None:
            print('\nTraining DPR from scratch')
            self.TrainFromScratch(cfg, corpus)
                
        # Build device and encoders (loaded or random init)
        self.device = torch.device(cfg.device)
        self.LoadModels()                    # loads saved weights if available
        doc_texts = [document.GetText() for document in self.index.GetDocuments()]
        self.doc_embeddings = self.EncodeDocuments(doc_texts)
    
    def TrainFromScratch(self, cfg, corpus):
        from src.training.trainer import Trainer
        from src.retrievers.models.encoder import TransformerEncoder
        from src.data_utils.dataloader import make_loader
        
        epochs = getattr(cfg.retriever, 'max_epochs', 5) or 5
        lr = getattr(cfg.retriever, 'lr', 1e-4) or 1e-4
        batch_size = getattr(cfg.retriever, 'batch_size', 32)
        
        enc_cfg = dict(
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            emb_dim=cfg.model.emb_dim,
            hidden_dim=cfg.model.hidden_dim,
            feed_forward_dim=cfg.model.feed_forward_dim,
            pooling_type=cfg.model.pooling_type,
        )
        
        weights_dir = os.path.dirname(self.query_weights_path)
        history = Trainer().train(
            TransformerEncoder(**enc_cfg),
            TransformerEncoder(**enc_cfg),
            make_loader(corpus.train, shuffle=True, batch_size=batch_size),
            make_loader(corpus.val, shuffle=True, batch_size=batch_size),
            epochs=epochs,
            lr=lr,
            device=cfg.device,
            weights_dir=weights_dir,
        )

        # used for the loss plot in results.ipynb
        with open(os.path.join(weights_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
        
    def LoadModels(self):
        """Build encoders and load pretrained weights if available."""
        m = self.cfg.model
        
        common_params = dict(
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            emb_dim=m.emb_dim,
            hidden_dim=m.hidden_dim,
            feed_forward_dim=m.feed_forward_dim,
            pooling_type=m.pooling_type,
        )
        
        query_encoder = TransformerEncoder(**common_params)
        document_encoder = TransformerEncoder(**common_params)
        query_encoder.to(self.device)
        document_encoder.to(self.device)
        
        # Load query encoder weights if file exists
        if os.path.exists(self.query_weights_path):
            state_dict = torch.load(self.query_weights_path, map_location=self.device, weights_only=True)
            query_encoder.load_state_dict(state_dict)

        # Load document encoder weights if file exists
        if os.path.exists(self.document_weights_path):
            state_dict = torch.load(self.document_weights_path, map_location=self.device, weights_only=True)
            document_encoder.load_state_dict(state_dict)

        # Set loaded encoders so they can be accessed by other methods / tests
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder

    def EncodeDocuments(self, doc_texts: list[str]) -> torch.Tensor:
        """Encode all corpus documents once (like TF-IDF's corpus matrix)."""
        max_length = self.cfg.retriever.document_max_length
        batch_size = getattr(self.cfg.retriever, 'batch_size', 32)
        
        all_embs = []
        self.document_encoder.eval()
        with torch.no_grad():
            for i in range(0, len(doc_texts), batch_size):
                batch = doc_texts[i:i + batch_size]
                ids = tokenize(batch, max_length).to(self.device)
                emb = self.document_encoder(ids)  # (batch, emb_dim)
                all_embs.append(emb.cpu())           # keep on CPU for storage
        
        return torch.cat(all_embs)  # (n_docs, emb_dim)

    def CalculateScores(self, queries: list[str]):
        """Encode queries and compute similarity with precomputed doc embeddings.
        
        Returns numpy array of shape (n_queries, n_docs), mirroring TF-IDF output.
        """
        max_length = self.cfg.retriever.query_max_length
        ids = tokenize(queries, max_length).to(self.device)
        self.query_encoder.eval()
        with torch.no_grad():
            score = self.query_encoder(ids).cpu() @ self.doc_embeddings.T
        return score.numpy()  # (n_queries, n_docs)
