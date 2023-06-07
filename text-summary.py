from newspaper import Article
from konlpy.tag import Kkma, Okt
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

class SentenceTokenizer(object):
    def __init__(self) -> None:
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = []
    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)

        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences
    
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences
    
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentences != '':
                nouns.append(' '.join([
                    noun for noun in self.twitter.nouns(str(sentence))
                    if noun not in self.stopwords and len(noun) > 1
                ]))
        return nouns

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return  self.graph_sentence
        
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}

class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로 
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}

class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
                    
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx =  self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        # index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary
    
    def summarize_v2(self):
        summary = []
        for idx in self.sorted_sent_rank_idx:
            summary.append((self.sentences[idx], self.sent_rank_idx[idx]))
        
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        
        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords



JVM_PATH = '/Library/Java/JavaVirtualMachines/zulu-17.jdk/Contents/Home/bin/java'

def main():
    okt = Okt(jvmpath=JVM_PATH)
    url = '''
업계는 증권성이 높아 규제 리스크가 큰 알트코인 자금이 증권성이 낮은 비트코인으로 쏠리면서 비트코인의 급격한 반등이 나타났다고 분석하고 있다. SEC는 가상자산거래소를 제소하면서 증권성이 높은 12개의 알트코인을 제시했는데 여기에는 BNB를 비롯해 ADA, MATIC, 솔라나(SOL), 샌드박스(SAND), 파일코인(FIL) 등이 포함됐다.
외환거래 플랫폼 오안다의 에드워드 모야 분석가는 “알트코인에 대한 SEC의 단속은 비트코인에 도움이 될 수 있다”며 “일부 투자자는 알트코인의 (투자) 포지션을 청산하고 비트코인 포지션을 다시 개설할 수 있다”고 내다봤다.
미 금융 당국의 가상자산 규제는 시장이 제도권 내에 편입하는 과정이라며 오히려 긍정적으로 평가하는 시선도 있다.
게리 겐슬러 SEC 위원장은 이날 트위터에 “코인베이스가 사기와 시세 조작을 방지하는 규제, 공시, 이익 상충에 대한 보호 조치, 일상적인 감독 등 투자자 보호 조치를 소홀히 했다”고 지적했다. 맷 호건 비트와이즈 자산운용 최고투자책임자는 “SEC의 제소가 단기적으로는 가상자산 시장의 불확실성을 높였지만 장기적으로는 큰 이득을 가져다줄 수 있다”고 평가했다.
가상자산이 마약 판매 수단으로 악용되고 있는 정황도 드러났다. 블록체인 분석기업 체이널리시스는 이날 ‘가상자산과 마약성 진통제’ 보고서를 통해 마약성 진통제인 펜타닐의 이동이 가상자산의 이동과 비슷하게 이뤄지고 있다는 분석을 내놨다.
미국 증권거래위원회(SEC)가 세계 최대 가상자산거래소 바이낸스와 미국 최대 거래소 코인베이스에 증권법 위반 혐의로 소송을 제기했다. 거래소가 취급하는 가상자산을 미등록 증권으로 본 것인데, 비트코인은 이 같은 악재에도 가격 변동이 크지 않았다.
7일 가상자산 중계사이트 코인마켓캡에 따르면 비트코인 가격은 이날 오후 3시 기준 3499만원으로 지난 5일 3505만원과 큰 차이가 없었다. 미 SEC가 지난 5일(현지시간) 바이낸스와 자오창펑 최고경영자(CEO)를 증권법 위반 혐의로 제소하고 다음 날 코인베이스를 같은 혐의로 제소했지만 큰 영향을 받지 않은 것이다. SEC의 바이낸스 제소 소식이 전해진 직후 비트코인 가격은 3318만원까지 6% 넘게 떨어졌으나 이날 오전 9시 3540만원까지 반등하는 저력을 과시했다. 반면 바이낸스코인(BNB), 카르다노(ADA), 폴리곤(MATIC) 등 알트코인(비트코인 이외 가상자산)은 1주일 전 가격보다 8∼10% 급락하며 약세를 보였다.
    '''
    textrank = TextRank(url)
    for row in textrank.summarize(3):
        print(row)
        print()
    print('keywords :',textrank.keywords())

if __name__ == '__main__':
    main()