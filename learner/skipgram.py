import numpy as np

from gensim.models import Word2Vec

PAD_TOKEN = "0"
SOS_TOKEN = "1"
EOS_TOKEN = "2"


def train_embeddings(config, data):
    print("Training embeddings...", end=" ")

    word2index = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 1,
        EOS_TOKEN: 2,
    }

    word_freqs = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 0,
        EOS_TOKEN: 0,
    }

    start_idx = len(word2index)

    w2v = Word2Vec(
        [s.split(" ") for s in data.fragments],
        size=config.embed_dim,
        window=config.get('embed_window'),
        min_count=1,
        negative=5,
        workers=20,
        iter=10,
        sg=1)

    vocab = w2v.wv.vocab
    word2index.update({k: v.index + start_idx for (k, v) in vocab.items()})
    word_freqs.update({k: v.count + start_idx for (k, v) in vocab.items()})

    tokens = np.random.uniform(-0.05, 0.05, size=(start_idx, config.embed_dim))
    embeddings = np.vstack([tokens, w2v[vocab]])
    path = config.path('config') / f'emb_{config.embed_dim}.dat'
    np.savetxt(path, embeddings, delimiter=",")

    print(f'Done.')

    return word2index, word_freqs
