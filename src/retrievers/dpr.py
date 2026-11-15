import os
import torch

from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
from src.data_utils.dataloader import tokenize


class DPR(Retriever):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._load_models()
        self.doc_embeddings = self._encode_documents()
    
    def _load_models(self):
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
        
        # Load query encoder weights if provided and file exists
        q_path = self.cfg.retriever.get('query_encoder_weights_path', None)
        if q_path is not None and os.path.exists(q_path):
            state_dict = torch.load(q_path, map_location=self.device, weights_only=True)
            query_encoder.load_state_dict(state_dict)

        # Load document encoder weights if provided and file exists
        d_path = self.cfg.retriever.get('document_encoder_weights_path', None)
        if d_path is not None and os.path.exists(d_path):
            state_dict = torch.load(d_path, map_location=self.device, weights_only=True)
            document_encoder.load_state_dict(state_dict)

        # Set loaded encoders so they can be accessed by other methods / tests
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder

    def _encode_documents(self) -> torch.Tensor:
        """Encode all corpus documents once (like TF-IDF's corpus matrix)."""
        doc_texts = [d.GetText() for d in self.index.GetDocuments()]
        batch_size = getattr(self.cfg.retriever, 'batch_size', 32)
        max_length = self.cfg.retriever.document_max_length
        
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
            score = self.query_encoder(ids) @ self.doc_embeddings.T
        return score.numpy()  # (n_queries, n_docs)


# --- Retriever integration -------------------------------------------------

    

        

