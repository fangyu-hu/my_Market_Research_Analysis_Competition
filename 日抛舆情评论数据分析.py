import re

import pandas as pd

import jieba
from snownlp import SnowNLP
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from emoji import replace_emoji


plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载停用词表
stopwords_file = 'stopwords.txt'
with open(stopwords_file, "r", encoding='utf-8') as words:
    stopwords = [i.strip() for i in words]


def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    text = replace_emoji(text)
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格

    # 去除停用词
    for word in stopwords:
        text = text.replace(word, '')
    return text.strip()


# 进行情感倾向分析
def analyze_sentiment(text):
    s = SnowNLP(text)
    return s.sentiments


# 绘制词云图
def generate_wordcloud(text):
    wordcloud = WordCloud(width=1000,
                          height=700,
                          background_color='white',  # 背景颜色
                          font_path='simhei.ttf',  # 字体
                          scale=15,  # 间隔
                          contour_width=5,  # 整个内容显示的宽度
                          contour_color='red',  # 内容显示的颜色 红色边境
                          ).generate(text)

    # wordcloud = WordCloud(font_path="simhei.ttf", background_color='white')
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# 绘制词频图
def plot_word_frequency(text):
    word_list = jieba.lcut(text)
    word_list = [word for word in word_list if word.strip()]
    word_counter = Counter(word_list)
    word_freq = word_counter.most_common(20)  # 取出现频率最高的前20个词语及其频次
    words, freqs = zip(*word_freq)

    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs)
    plt.xticks(rotation=45)
    plt.xlabel('词语')
    plt.ylabel('频次')
    plt.title('评论词语频次图')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('日抛小红书评论.csv', encoding="utf-8-sig")  # 读取评论文件，修改成对应的文件名

    comments = data['评论内容']
    location = data['IP属地']
    like_count = data['点赞数量']
    print(like_count.describe())

    print(location.value_counts())
    location.value_counts()[:15].plot(kind='pie', autopct='%.1f%%')
    plt.show()

    # 去除重复的评论
    comments = comments.drop_duplicates()
    print(comments.head())

    # 数据清洗：去除停用词、符号、标签
    clean_comments = []
    for comment in comments:
        comment = clean(str(comment))
        clean_comments.append(comment)

    for com in clean_comments[0:5]:
        print(com)

    # 对评论进行分词,去除一些停用词
    segmented_comments = []
    for comment in clean_comments:
        if len(comment) == 0:
            continue
        seg_list = ' '.join(jieba.lcut(comment, cut_all=True))
        segmented_comments.append(seg_list)

    print(segmented_comments)

    # 分析每条评论的情感倾向得分
    sentiment_scores = [analyze_sentiment(comment) for comment in segmented_comments]

    # 绘制情感分直方图
    bins = np.arange(0, 1.1, 0.1)
    plt.hist(sentiment_scores, bins, color='#4F94CD', alpha=0.9)
    plt.xlim(0, 1)
    plt.xlabel('情感分')
    plt.ylabel('数量')
    plt.title('情感分直方图')
    plt.show()

    # 根据情感倾向分数将评论分类为积极和消极
    positive_comments = [comment for comment, score in zip(segmented_comments, sentiment_scores) if score > 0.5]
    negative_comments = [comment for comment, score in zip(segmented_comments, sentiment_scores) if score <= 0.5]

    # 积极消极评论占比
    pie_labels = ['积极评论', '消极评论']
    plt.pie([len(positive_comments), len(negative_comments)],
            labels=pie_labels, autopct='%1.2f%%', shadow=True)
    plt.title("积极和消极评论占比")
    plt.show()

    # 绘制积极评论词云图
    positive_text = ' '.join(positive_comments)
    generate_wordcloud(positive_text)

    # 绘制消极评论词云图
    negative_text = ' '.join(negative_comments)
    generate_wordcloud(negative_text)

    # 绘制总的词频图
    total_text = ' '.join(clean_comments)
    plot_word_frequency(total_text)
