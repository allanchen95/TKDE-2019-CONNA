import logging
import sys
sys.path.append("..")
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils import data_utils
from utils.data_utils import Singleton
from utils import settings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@Singleton
class EmbeddingModel:

    def train(self):
        features = data_utils.load_data('human_labeled_data/', "pub.features")
        index = 0
        author_data = []
        word_data = []
        for pub_index in range(len(features)):
            pub_features = features[pub_index]
            if (pub_features == None):
                continue
            for author_index in range(len(pub_features)):
                aid, author_features, word_features = pub_features[author_index]

                if index % 100000 == 0:
                    print(index, author_features, word_features)
                index += 1

                random.shuffle(author_features)
                author_data.append(author_features)
                random.shuffle(word_features)
                word_data.append(word_features)

        self.author_model = Word2Vec(
            author_data, size=settings.EMB_DIM, window=5, min_count=5, workers=20,
        )
        self.author_model.save(join('Essential_Embeddings/emb/', 'author_name.emb'))
        self.word_model = Word2Vec(
            word_data, size=settings.EMB_DIM, window=5, min_count=5, workers=20,
        )
        self.word_model.save(join('Essential_Embeddings/emb/', 'word.emb'))

    def load_author_name_emb(self):
        self.author_model = Word2Vec.load(join('Essential_Embeddings/emb/', 'author_name.emb'))
        return self.author_model

    def load_word_name_emb(self):
        self.word_model = Word2Vec.load(join('Essential_Embeddings/emb/', 'word.emb'))
        return self.word_model


if __name__ == '__main__':
    emb_model = EmbeddingModel.Instance()
    emb_model.train()
    print('Train finished.')
