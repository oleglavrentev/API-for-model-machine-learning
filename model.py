from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import words


class Assist():

    def recognize(self, data):
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(list(words.data_set.keys()))
        clf = LogisticRegression()
        clf.fit(vectors, list(words.data_set.values()))
        del words.data_set
        # получаем вектор полученного текста
        # сравниваем с вариантами, получая наиболее подходящий ответ
        text_vector = vectorizer.transform([data]).toarray()[0]
        answer = clf.predict([text_vector])[0]

        return answer

