# word_cloud.py

import pandas as pd
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import networkx as nx

class Action:
    def __init__(self, file_name):
        self.file_name = file_name

    def analyze(self):
        data = pd.DataFrame()
        for file in self.file_name:
            data = data.append(pd.read_csv(file), ignore_index=True)
        # 对项目名称下的值进行处理，将其转化为str类型
        data['项目名称'] = data['项目名称'].astype(str)
        data_cut = data['项目名称'].apply(jieba.lcut)

        with open('stoplist.txt', 'r', encoding='utf-8') as f:
            stop_list = f.read().splitlines()
        stop_list.extend(["[", "]", ",", " "])  # 修正停用词列表
        data_after_stop = data_cut.apply(lambda x: [i for i in x if i not in stop_list])
        data_after_stop.to_csv('data_after_stop.csv', index=False, encoding='utf-8-sig')

        data_after_stop_flaten = []
        for i in data_after_stop:
            data_after_stop_flaten.extend(i)
        word_freq = pd.Series(data_after_stop_flaten).value_counts()
        
        # 创建知识图谱
        self.create_knowledge_graph(word_freq)
        
        return word_freq

    def create_knowledge_graph(self, word_freq):
        # 创建图结构
        G = nx.Graph()
        for word, freq in word_freq.items():
            G.add_node(word, size=freq)
        
        # 这里可以根据需要定义边的创建规则
        for word1 in word_freq.index:
            for word2 in word_freq.index:
                if word1 != word2 and self.some_similarity_function(word1, word2) > 0.5:
                    G.add_edge(word1, word2)
        
        # 保存知识图谱
        nx.write_gml(G, 'knowledge_graph.gml')

    def some_similarity_function(self, word1, word2):
        # 示例：基于某种规则计算词之间的相似度
        # 这里可以使用文本相似度算法，如余弦相似度等
        return 0.5  # 示例值，实际应根据需要计算

class App:
    def __init__(self, word_freq):
        self.word_freq = word_freq

    def draw(self):
        # 绘制词云图
        mask = plt.imread('InternLM.jpg')
        wc = WordCloud(font_path='./simkai.ttf', mask=mask, background_color='white',
                       max_words=500,
                       max_font_size=150,
                       relative_scaling=0.6,
                       random_state=50,
                       scale=2
                       )
        wc.fit_words(self.word_freq)
        image_color = ImageColorGenerator(mask)
        wc.recolor(color_func=image_color)

        plt.figure(figsize=(20, 20))
        plt.imshow(wc, interpolation='bicubic')
        plt.axis('off')
        plt.show()

    def draw_knowledge_graph(self):
        # 读取知识图谱
        G = nx.read_gml('knowledge_graph.gml')

        # 绘制知识图谱
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # 布局
        sizes = [G.nodes[node]['size'] * 10 for node in G.nodes]  # 节点大小
        nx.draw(G, pos, with_labels=True, node_size=sizes, font_size=12, node_color='lightblue', edge_color='gray')
        plt.title('Knowledge Graph')
        plt.show()

if __name__ == '__main__':
    file_name = ['InternLM_all.csv']  # 文件名
    # 实例化Action对象
    action = Action(file_name)
    word_freq = action.analyze()
    # 实例化App对象
    app = App(word_freq)
    app.draw()
    app.draw_knowledge_graph()
