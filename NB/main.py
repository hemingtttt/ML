import copy
import os.path
import string
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 加载停用词
stopwords = set(nltk.corpus.stopwords.words('english'))
# 文本类别
text_type = {'World': 0, 'Sci/Tech': 1, 'Sports': 2, 'Business': 3}


def load(path, type_r):
    """
    加载数据并进行预处理
    :param path: 数据路径
    :param type_r: 使用的单词还原方法
    """
    x, y = [], []
    file = open(path, 'r')
    trans = str.maketrans('', '', string.punctuation)
    if type_r == 'stemmer':
        re = nltk.stem.porter.PorterStemmer().stem
    elif type_r == 'lemmatizer':
        re = nltk.stem.WordNetLemmatizer().lemmatize
    else:
        raise ValueError('type error')
    for line in file:
        temp = line.split('|')  # 将类别和文本分离开
        # 预处理文本
        sent = temp[1].strip().lower()  # 全小写
        sent = sent.translate(trans)  # 去除标点符号
        sent = nltk.word_tokenize(sent)  # 将文本标记化
        sent = [s for s in sent if not ((s in stopwords) or s.isdigit())]  # 去停用词和数字
        sent = [re(s) for s in sent]  # 还原: 词干提取/词形还原
        x.append(sent)
        # 预处理类别
        y.append(text_type[temp[0].strip()])
    file.close()
    return x, y


def words2dic(sent):
    """
    生成数据对应的文本库id
    :param sent: 数据集
    :return: 文本库
    """
    dicts = {}
    i = 0
    for words in sent:
        for word in words:
            if word not in dicts:
                dicts[word] = i
                i += 1
    return dicts


def train_TF(data_x, data_y):
    """
    朴素贝叶斯(TF方法)训练
    :param data_x: 训练数据
    :param data_y: 真值
    """
    # 构建词典，用于生成统计矩阵
    dicts = words2dic(data_x)
    # n(w_i in w_c) 创建词频矩阵
    word_fre = np.zeros((len(dicts), 4), dtype=np.int32)
    # n(c, text) 每类下的句总数
    sent_fre = np.zeros((1, 4), dtype=np.int32)
    # 更新矩阵
    for x, y in zip(data_x, data_y):
        for word in x:
            if word in dicts:  # 过滤未登录词
                word_fre[dicts[word], y] += 1
        sent_fre[0, y] += 1

    # 计算P(c)：类别c的文档数 / 总文档数
    total_docs = len(data_y)
    p_c = sent_fre[0] / total_docs

    # 计算P(w_i|c)，并加入拉普拉斯平滑
    # 拉普拉斯平滑：分子+1，分母+V（V为词汇表大小）
    V = len(dicts)
    p_stage = np.zeros((V, 4), dtype=np.float32)
    for c in range(4):
        # 类别c的总词数 + V（每个词加1，共V个词）
        denom = np.sum(word_fre[:, c]) + V
        for w_idx in range(V):
            # 词w在类别c中的出现次数 + 1
            numer = word_fre[w_idx, c] + 1
            p_stage[w_idx, c] = numer / denom

    return dicts, p_stage, p_c


def test_TF(data_x, data_y, dicts, p_stage, p_c):
    """
    测试TF方法的准确率
    """
    # 计算ln P(c)
    ln_p_c = np.log(p_c)
    # 计算ln P(w_i|c)
    ln_p_s = np.log(p_stage)
    # 计算准确率
    count = 0
    for x, y_true in zip(data_x, data_y):
        # 初始化每个类别的对数概率为ln P(c)
        ln_p = ln_p_c.copy()
        for word in x:
            if word in dicts:  # 过滤未收录词
                w_idx = dicts[word]
                # 累加每个词的对数概率
                ln_p += ln_p_s[w_idx]
        # 获取概率最大的类别
        y_pred = np.argmax(ln_p)
        if y_pred == y_true:
            count += 1
    print('Accuracy: {}/{} {:.2f}%'.format(count, len(data_y), 100 * count / len(data_y)))


def train_B(data_x, data_y):
    """
    伯努利方法训练
    """
    # 构建词典
    dicts = words2dic(data_x)
    V = len(dicts)
    # 统计每个类别中包含某个词的文档数
    word_in_class = np.zeros((V, 4), dtype=np.int32)
    # 每类下的句总数
    class_docs = np.zeros(4, dtype=np.int32)

    # 遍历每个文档
    for x, y in zip(data_x, data_y):
        class_docs[y] += 1
        # 记录当前文档中出现的词（去重）
        words_in_doc = set(x)
        for word in words_in_doc:
            if word in dicts:
                w_idx = dicts[word]
                word_in_class[w_idx, y] += 1  # 只要出现过就+1

    # 计算P(c)
    total_docs = len(data_y)
    p_c = class_docs / total_docs

    # 计算P(w_i|c)，伯努利方法使用词是否出现的概率
    # 拉普拉斯平滑：分子+1，分母+2（二项式分布平滑）
    p_stage = np.zeros((V, 4), dtype=np.float32)
    for c in range(4):
        for w_idx in range(V):
            # 分子：类别c中包含词w的文档数 + 1
            numer = word_in_class[w_idx, c] + 1
            # 分母：类别c的文档数 + 2（+1 for 出现，+1 for 未出现）
            denom = class_docs[c] + 2
            p_stage[w_idx, c] = numer / denom

    return dicts, p_stage, p_c


def test_B(data_x, data_y, dicts, p_stage, p_c):
    """
    测试伯努利方法的准确率
    """
    # 计算ln P(c)
    ln_p_c = np.log(p_c)
    # 计算ln P(w_i|c)和ln(1-P(w_i|c))
    ln_p_w = np.log(p_stage)
    ln_p_not_w = np.log(1 - p_stage)
    # 计算准确率
    count = 0
    for x, y_true in zip(data_x, data_y):
        # 初始化每个类别的对数概率为ln P(c)
        ln_p = ln_p_c.copy()
        words_in_doc = set(x)
        for w_idx in range(len(dicts)):
            word = next((k for k, v in dicts.items() if v == w_idx), None)
            if word in words_in_doc:
                # 词在文档中出现，累加ln P(w_i|c)
                ln_p += ln_p_w[w_idx]
            else:
                # 词未出现，累加ln(1-P(w_i|c))
                ln_p += ln_p_not_w[w_idx]
        # 获取概率最大的类别
        y_pred = np.argmax(ln_p)
        if y_pred == y_true:
            count += 1
    print('Accuracy: {}/{} {:.2f}%'.format(count, len(data_y), 100 * count / len(data_y)))


if __name__ == '__main__':
    '''超参数设置'''
    # 单词还原方法
    type_re = ['stemmer', 'lemmatizer'][0]
    # 训练方法
    type_train = ['TF', 'Bernoulli'][0]
    print('训练方法: {}'.format(type_train))
    print('还原方法: {}'.format(type_re))

    '''读取训练数据并进行预处理'''
    train_x, train_y = load('./data/news_category_train_mini.csv', type_re)
    test_x, test_y = load('./data/news_category_test_mini.csv', type_re)
    print('load success')

    '''开始训练'''
    if type_train == 'TF':
        dictionary, p_stage, p_c = train_TF(train_x, train_y)
    elif type_train == 'Bernoulli':
        dictionary, p_stage, p_c = train_B(train_x, train_y)

    '''计算准确率'''
    print("训练集准确率:")
    if type_train == 'TF':
        test_TF(train_x, train_y, dictionary, p_stage, p_c)
        print("测试集准确率:")
        test_TF(test_x, test_y, dictionary, p_stage, p_c)
    elif type_train == 'Bernoulli':
        test_B(train_x, train_y, dictionary, p_stage, p_c)
        print("测试集准确率:")
        test_B(test_x, test_y, dictionary, p_stage, p_c)

