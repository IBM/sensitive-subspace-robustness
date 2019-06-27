import numpy as np
from sklearn.decomposition import TruncatedSVD

from utils import load_data, load_test_names, load_nyc_names, print_summary
from SenSR import train_nn, train_fair_nn

# Download (and unpack) positive and negative words from
# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
# and put them into
data_path = './sentiment/'

# Download Common Crawl (42B tokens) GloVe word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './embeddings/glove.42B.300d.txt'

# Download Popular Baby Names in CSV format from
# https://catalog.data.gov/dataset/most-popular-baby-names-by-sex-and-mothers-ethnic-group-new-york-city-8c742
# and put them into
nyc_names_path = './nyc_names/Popular_Baby_Names.csv'


## Load data and embeddings
# Loading GloVe might take couple of minutes
embeddings, X_train, X_test, y_train, y_test, train_vocab, test_vocab = load_data(data_path, embeddings_path)

## Load test names and their embeddings
test_df, test_names_embed = load_test_names(embeddings)

## Load Popular NYC Baby Names and their embeddings
nyc_names_embed = load_nyc_names(nyc_names_path, embeddings)


## Baseline
baseline_weights, _, baseline_test_logits = train_nn(X_train, y_train, X_test=X_test, y_test=y_test, epoch=2000, batch_size=1000)
baseline_accuracy = (baseline_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, baseline_names_logits, _ = train_nn(test_names_embed, y_train=None, weights=baseline_weights, epoch=0)
test_df['baseline_logits'] = baseline_names_logits[:,1] - baseline_names_logits[:,0]
print_summary(test_df, 'baseline', baseline_accuracy)

## SenSR_0 expert
expert_sens_directions = np.copy(test_names_embed)
sensr0_expert_weights, _, sensr0_expert_test_logits = train_fair_nn(X_train, y_train, expert_sens_directions, X_test = X_test, y_test=y_test)
sensr0_expert_accuracy = (sensr0_expert_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, sensr0_expert_names_logits, _ = train_nn(test_names_embed, y_train=None, weights=sensr0_expert_weights, epoch=0)
test_df['sensr0_expert_logits'] = sensr0_expert_names_logits[:,1] - sensr0_expert_names_logits[:,0]
print_summary(test_df, 'sensr0_expert', sensr0_expert_accuracy)

## SenSR_0
# Learning sensitive direction from Popular Baby Names
tSVD = TruncatedSVD(n_components=50)
tSVD.fit(nyc_names_embed)
svd_sens_directions = tSVD.components_


sensr0_weights, _, sensr0_test_logits = train_fair_nn(X_train, y_train, svd_sens_directions, X_test = X_test, y_test=y_test)
sensr0_accuracy = (sensr0_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, sensr0_names_logits, _ = train_nn(test_names_embed, y_train=None, weights=sensr0_weights, epoch=0)
test_df['sensr0_logits'] = sensr0_names_logits[:,1] - sensr0_names_logits[:,0]
print_summary(test_df, 'sensr0', sensr0_accuracy)

## SenSR
sensr_weights, _, sensr_test_logits = train_fair_nn(X_train, y_train, svd_sens_directions, X_test = X_test, y_test=y_test, full_step=0.01, eps=0.1)
sensr_accuracy = (sensr_test_logits.argmax(axis=1)==y_test.argmax(axis=1)).mean()
_, sensr_names_logits, _ = train_nn(test_names_embed, y_train=None, weights=sensr_weights, epoch=0)
test_df['sensr_logits'] = sensr_names_logits[:,1] - sensr_names_logits[:,0]
print_summary(test_df, 'sensr', sensr_accuracy)
