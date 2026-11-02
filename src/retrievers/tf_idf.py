from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        text = text.lower()
        # Everything that is not a letter or a digit becomes a space
        text = ''.join(char if char.isalnum() else ' ' for char in text)
        return text.split()

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return list(set(self.PreprocessText(query)))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return list(set(self.PreprocessText(document.GetText())))

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        document_frequencies = {}
        for document in self.index.GetDocuments():
            for term in self.GetDocumentVocabulary(document):
                document_frequencies[term] = document_frequencies.get(term, 0) + 1
        return document_frequencies

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        # idf(t) = log(N / df(t))
        n_documents = len(self.index.GetDocuments())
        document_frequencies = self.GetDocumentFrequencies()
        return {term: np.log(n_documents / df) for term, df in document_frequencies.items()}

    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        term_counts = {}
        for term in self.PreprocessText(document.GetText()):
            term_counts[term] = term_counts.get(term, 0) + 1
        return term_counts

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # tf(t, d) = (times t occurs in d) / (number of words in d)
        # tf-idf(t, d) = tf(t, d) * idf(t)
        for i, document in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(document)
            document_length = sum(term_counts.values())
            term_indices = [self.term_to_index[term] for term in term_counts]
            tfidf_values = [count / document_length * self.idf[term] for term, count in term_counts.items()]
            tfidf_matrix[i, term_indices] = tfidf_values  

        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        Args:
            query (str): The query to convert
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for i, query in enumerate(queries):
            for term in self.PreprocessText(query):
                # Words that never occur in the corpus cannot match any document, so they are skipped
                if term in self.term_to_index:
                    query_vector[i, self.term_to_index[term]] += 1

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    