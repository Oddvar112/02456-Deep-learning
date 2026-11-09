from src.data_utils.query import Query


def calculatePrecision(relevancies: list[bool]):
    return sum(relevancies) / len(relevancies)


def calculateReciprocalRank(relevancies: list[bool]):
    for i, relevant in enumerate(relevancies):
        if relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculateRecall(relevancies: list[bool], query: Query): #Capped recall@k takes smalled: k or "total relevant docs, so its posible to achive 100% even when k is bigger than num of all relevant docks"
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None, 'precision': None, 'mrr': None}
    n_queries = len(queries)

    total_recall = 0
    total_precision = 0
    total_mrr = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i])
        total_mrr += calculateReciprocalRank(relevancies[i])
    

    scores['recall'] = total_recall / n_queries
    scores['precision@k'] = total_precision / n_queries
    scores['mrr'] = total_mrr / n_queries
    return scores
