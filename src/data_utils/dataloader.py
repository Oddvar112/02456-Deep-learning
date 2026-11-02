from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenize(texts: list[str], max_length: int):
    """
    Turns a list of texts into a tensor of token ids, one row per text.
    """
    return tokenizer(texts,
                     padding='max_length',   # fill short texts with 0 up to max_length
                     truncation=True,        # cut texts that are longer than max_length
                     max_length=max_length,  # length of every row
                     return_tensors='pt'     # return a PyTorch tensor instead of Python lists
                     )['input_ids']          # keep only the token ids 


def make_loader(dataset_split, shuffle: bool, batch_size: int = 32):
    """
    Turns a dataset split into batches of (question, answer) pairs.
    """
    questions, answers = [], []
    for row in dataset_split:                 
        for text in row['text']:
            questions.append(row['question'])
            answers.append(text)

    dataset = TensorDataset(tokenize(questions, 64),tokenize(answers, 256))    

    return DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)       
