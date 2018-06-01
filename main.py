#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import itchat
import numpy as np
import jieba.analyse
import matplotlib.pyplot as plt

from PIL import Image
from pylab import mpl
from snownlp import SnowNLP
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator


# 让Matplotlib支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 去除轮廓线
mpl.rcParams['patch.edgecolor'] = 'FFFFFF'

DEFAULT_DPI = 300
FIG_SIZE = (12, 7)

base_path = os.path.dirname(__file__)

class WeChatAnalyse(object):
    def __init__(self):
        self.friends = []

    def get_wechat_friends(self):
        itchat.auto_login(hotReload=True, enableCmdQR=2)
        self.friends = itchat.get_friends(True)
        self.cur_user_name = self.friends[0]['NickName']

    def analyse_sex(self):
        sexs = list(map(lambda x:x['Sex'], self.friends[1:]))
        counts = list(map(lambda x:x[1], Counter(sexs).items()))
        labels = [u'未知', u'男性', u'女性']
        colors = ['red', 'yellowgreen', 'lightskyblue']
        plt.figure(figsize=FIG_SIZE, dpi=DEFAULT_DPI)
        plt.axes(aspect=1)
        plt.pie(counts, labels=labels, colors=colors, labeldistance=1.1, explode=[0.03, 0.03, 0.03],
                autopct='%0.f%%', shadow=False, startangle=90, pctdistance=0.6, rotatelabels=True)
        plt.legend(loc='upper right')
        plt.title(u'%s的微信一共%d好友' % (self.cur_user_name, len(sexs)))
        plt.savefig(os.path.join(base_path, 'output', 'sex.png'))
        plt.close()

    def get_friend_head_image(self):
        base_folder = os.path.join(base_path, 'HeadImages')
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        for i in range(1, len(self.friends)):
            friend = self.friends[i]
            user_name = friend['UserName']
            nick_name = friend['NickName']
            img_file = os.path.join(base_folder, "%s.jpg" % nick_name)
            img_data = itchat.get_head_img(userName=user_name)
            if not os.path.exists(img_file):
                try:
                    with open(img_file, 'wb+') as fp:
                        fp.write(img_data)
                except IOError:
                    print u"%s头像下载错误" % nick_name

    def analyse_sign(self):
        signatures = ""
        emotions = []
        for friend in self.friends:
            sign = friend['Signature']
            if sign:
                sign = sign.strip().replace('span', '').replace('class', '').replace('emoji', '')
                sign = re.sub(r'1f(\d.+)', '', sign)
                if sign:
                    nlp = SnowNLP(sign)
                    emotions.append(nlp.sentiments)
                    signatures += ' '.join(jieba.analyse.extract_tags(sign, 6))

        with open('signatures.txt', 'w+') as fp:
            fp.write(signatures.encode('utf-8'))

        # 词云
        back_coloring = np.array(Image.open(os.path.join(base_path, 'res', 'flower.jpg')))
        wc = WordCloud(font_path=os.path.join(base_path, 'res', 'SourceHanSerifK-Light.otf'), 
                            background_color='white',
                            max_words=1200, mask=back_coloring, max_font_size=90,
                            random_state=45, margin=2)
        wc.generate(signatures)
        plt.figure(figsize=FIG_SIZE, dpi=DEFAULT_DPI)
        plt.title(u'%s的微信好友签名关键字分析' % self.cur_user_name)
        plt.xlabel(u'好友签名')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(base_path, 'output', 'signatures.png'))
        plt.close()

        # 签名情感分析
        count_good = len(list(filter(lambda x:x>0.66, emotions)))
        count_normal = len(list(filter(lambda x:x>=0.33 and x<=0.66, emotions)))
        count_bad = len(list(filter(lambda x:x<0.33, emotions)))

        labels = [u'负面消极', u'中性', u'正面积极']
        values = (count_bad, count_normal, count_good)
        plt.figure(figsize=FIG_SIZE, dpi=DEFAULT_DPI)
        plt.ylabel(u'频数')
        plt.xticks(range(3), labels)
        plt.legend(loc='upper right')
        plt.bar(range(3), values, color='rgb')
        plt.title(u'%s的微信好友签名信息情感分析' % self.cur_user_name)
        plt.savefig(os.path.join(base_path, 'output', 'qg.png'))
        plt.close()

    def analyse_city(self):
        citys = list(map(lambda x:x['City'], self.friends[1:]))
        citys = list(map(lambda x:u'未知' if x == '' else x, citys))

        city_labels = []
        city_values = []
        for c, n in Counter(citys).items():
            city_labels.append(c)
            city_values.append(n)

        plt.figure(figsize=FIG_SIZE, dpi=DEFAULT_DPI)
        plt.ylabel(u'人数')
        x_values = range(len(city_labels))
        plt.xticks(x_values, city_labels)
        for a,b in zip(x_values, city_values):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        plt.legend(loc='upper right')
        plt.bar(x_values, city_values, color='r')
        plt.title(u'%s的微信好友城市分布' % self.cur_user_name)
        plt.gcf().autofmt_xdate()
        plt.savefig(os.path.join(base_path, 'output', 'city.png'))
        plt.close()

    def analyse_province(self):
        provinces = list(map(lambda x:x['Province'], self.friends[1:]))
        provinces = list(map(lambda x:u'未知' if x == '' else x, provinces))

        city_labels = []
        city_values = []
        for c, n in Counter(provinces).items():
            city_labels.append(c)
            city_values.append(n)

        plt.figure(figsize=FIG_SIZE, dpi=DEFAULT_DPI)
        plt.ylabel(u'人数')
        x_values = range(len(city_labels))
        plt.xticks(x_values, city_labels)
        for a,b in zip(x_values, city_values):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        plt.legend(loc='upper right')
        plt.bar(x_values, city_values, color='r')
        plt.title(u'%s的微信好友省份分布' % self.cur_user_name)
        plt.gcf().autofmt_xdate()
        plt.savefig(os.path.join(base_path, 'output', 'province.png'))
        plt.close()



def main():
    weChat = WeChatAnalyse()
    weChat.get_wechat_friends()
    weChat.analyse_sex()
    weChat.analyse_sign()
    weChat.analyse_city()
    weChat.analyse_province()

if __name__ == '__main__':
    main()