from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    # K1: The saturation parameter
    # B: Penalties for long documents 
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        
        # Get the number of documents
        self.number_of_documents = len(self.index.GetDocuments())

        # Map the terms to corresponding index in the matrix 
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Get the document frequencies for each term
        self.document_frequencies = self.GetDocumentFrequencies()

        # Get updated IDF folowing the formula in the description
        self.idf = self.GetUpdatedIDF()

        # Compute the BM25 matrix
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

        # Create matrix from scipy parse 
        matrix = lil_matrix((self.number_of_documents, len(self.corpus_vocabulary)))

        # Get values from GetDocumentLengths method
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()

        for row, document in enumerate(documents):
            # Get the term counts for the document
            countTerm = self.GetDocumentTermCounts(document)

            # Calculate the BM25 score for each term in the document and store it in the matrix
            for term, termFrequency in countTerm.items():
                # Looks up the index of the term in the matrix
                column = self.term_to_index[term]
                # BM25 IDF from the assignment log((N + 1) / (n + 0.5))
                idf = np.log((number_of_documents + 1) / (self.document_frequencies[term] + 0.5))
                # BM25 formular: IDF(q) * ( f(q,d) * (k1 + 1) ) / ( f(q,d) + k1 * (1 - b + b * DocumentLength / avarageDocumentLength))
                matrix[row, column] = idf *(termFrequency * (self.k1 + 1)) / (termFrequency + self.k1 * (1 - self.b + (self.b * self.document_lengths[document] / self.average_document_length)))

        return matrix

    def GetUpdatedIDF(self):

        # Calculate the updated IDF for BM25 using this formula: IDF(q_i) = log( (N + 1) / (n(q_i) + 0.5) )
        idf = {term: np.log((self.number_of_documents + 1) / (df + 0.5)) for term, df in self.document_frequencies.items()}
        
        return idf
    
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