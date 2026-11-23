import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
    
    ### Initialize TF-IDF class
    tf_idf = TFIDF(documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries
    tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)
    
    ### Evaluate TF-IDF class
    tfidf_train_scores = Evaluate(tfidf_train_relevancies, train_queries)
    tfidf_val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    tfidf_test_scores = Evaluate(tfidf_test_relevancies, test_queries)
    
    print(f'TF-IDF Train results: {tfidf_train_scores}')
    print(f'TF-IDF Val results: {tfidf_val_scores}')
    print(f'TF-IDF Test results: {tfidf_test_scores}')


    ### Initialize BM25 class
    bm25 = BM25(documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries using BM25
    bm25_train_relevancies = retrieveQueryAndGetRelevancies(bm25, train_queries, k=cfg.top_k)
    bm25_val_relevancies = retrieveQueryAndGetRelevancies(bm25, val_queries, k=cfg.top_k)
    bm25_test_relevancies = retrieveQueryAndGetRelevancies(bm25, test_queries, k=cfg.top_k)
    
    ### Evaluate BM25
    bm25_train_scores = Evaluate(bm25_train_relevancies, train_queries)
    bm25_val_scores = Evaluate(bm25_val_relevancies, val_queries)
    bm25_test_scores = Evaluate(bm25_test_relevancies, test_queries)
    
    print(f'BM25 Train results: {bm25_train_scores}')
    print(f'BM25 Val results: {bm25_val_scores}')
    print(f'BM25 Test results: {bm25_test_scores}')

    
    ### Train and evaluate one DPR model for each pooling type
    for pooling in ['cls', 'mean', 'max']:
        cfg.model.pooling_type = pooling
        # Each model gets its own folder
        cfg.retriever.weights_dir = f'outputs/weights_{pooling}_b{cfg.retriever.batch_size}'

        ### Initialize DPR class
        dpr = DPR(cfg=cfg, corpus=corpus)

        dpr_train_relevancies = retrieveQueryAndGetRelevancies(dpr, train_queries, k=cfg.top_k)
        dpr_val_relevancies   = retrieveQueryAndGetRelevancies(dpr, val_queries, k=cfg.top_k)
        dpr_test_relevancies  = retrieveQueryAndGetRelevancies(dpr, test_queries, k=cfg.top_k)

        dpr_train_scores     = Evaluate(dpr_train_relevancies, train_queries)
        dpr_val_scores       = Evaluate(dpr_val_relevancies, val_queries)
        dpr_test_scores      = Evaluate(dpr_test_relevancies, test_queries)

        print(f'DPR {pooling} Train results: {dpr_train_scores}')
        print(f'DPR {pooling} Val results: {dpr_val_scores}')
        print(f'DPR {pooling} Test results: {dpr_test_scores}')

if __name__ == '__main__':
    main()