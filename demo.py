from lda import LDA
from dataset import TwentyNewsDataset
import time
dataset = TwentyNewsDataset()
dataset.load_data()
n_topics = 20

lda = LDA(n_topics)
lda.initialize(dataset.data_matrix)
lda.load_label('labels.txt', dataset.dictionary)
print(lda.print_labels())

for _ in range(100):
    lda.fit()
    
lda.get_topic_word()
lda.get_doc_topic()
lda.print_top_words(dataset.dictionary, 10)