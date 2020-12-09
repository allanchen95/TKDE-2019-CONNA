import nltk
from nltk.corpus import stopwords

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')


stemmer = nltk.stem.PorterStemmer()
stop = stopwords.words('english')
stop.extend(['-', 'at'])


def stem(word):
    return stemmer.stem(word)


def clean_sentence(text, stemming=False):
    for token in punct:
        text = text.replace(token, "")
    words = text.split()
    processed_words = []
    for w in words:
        w = w.lower()
        if w in stop:
            continue
        if stemming:
            w = stem(w)
        processed_words.append(w)
    words = processed_words
    return words


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)
