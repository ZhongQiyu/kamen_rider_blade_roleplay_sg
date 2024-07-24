import pandas as pd
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

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
        return word_freq

class App:
    def __init__(self, word_freq):
        self.word_freq = word_freq

    def draw(self):
        # 绘制词云图
        mask = plt.imread('InternLM.jpg')
        # 创建词云图对象
        wc = WordCloud(font_path='./simkai.ttf', mask=mask, background_color='white',
                       max_words=500,
                       max_font_size=150,  # 减小最大字体大小
                       relative_scaling=0.6,  # 设置字体大小与词频的关联程度为0.4
                       random_state=50,
                       scale=2  # 增加 scale 值
                       )  # font_path的相对路径
        # 加载词频
        wc.fit_words(self.word_freq)
        image_color = ImageColorGenerator(mask)  # 设置生成词云的颜色，如去掉这两行则字体为默认颜色
        wc.recolor(color_func=image_color)

        # plt 绘出词云图
        plt.figure(figsize=(20, 20))  # 调整图片大小
        plt.imshow(wc, interpolation='bicubic')  # 使用双三次插值
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    file_name = ['InternLM_all.csv']  # 文件名
    # 实例化Action对象
    action = Action(file_name)
    word_freq = action.analyze()
    # 实例化APP对象
    app = App(word_freq)
    app.draw()
