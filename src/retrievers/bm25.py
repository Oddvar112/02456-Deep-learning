from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    # K1: When repating words stop adding to the score 
    # B: Penalties for long documents 
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.ComputeBM25Matrix()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def ComputeBM25Matrix(self):
        documents = self.index.GetDocuments()
        number_of_documents = len(documents)

        # Create matrix from scipy parse 
        matrix = lil_matrix((number_of_documents, len(self.corpus_vocabulary)))

        # Count the terms 
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Get values from GetDocuemtLengths method
        self.document_frequencies = self.GetDocumentFrequencies()
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()

        for row, document in enumerate(documents):
            countTerm = self.GetDocumentTermCounts(document)
            for term, termFrequency in countTerm.items():
                column = self.term_to_index[term]
                # BM25 formular: IDF(q) * ( f(q,D) * (k1 + 1) ) / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl))
                matrix[row, column] = self.getIDF(term) * (termFrequency * (self.k1 + 1)) / (termFrequency + self.k1 * (1 - self.b + (self.b * self.document_lengths[document] / self.average_document_length)))

        return matrix.tocsr()

    def getIDF(self, term: str) -> float:
        # Formular: IDF(q) = log((index - freq(q) + 0.5) / (freq(q) + 0.5) + 1 )
        DocumentFreq = self.document_frequencies.get(term, 0)
        return np.log((len(self.index.GetDocuments()) - DocumentFreq + 0.5) / (DocumentFreq + 0.5) + 1)

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores