import numpy as np



class SVM:
    def __init__(self, features: int):
        """
        初始化svm的初值
        Args:
            features: 输入数据的维度
        """
        self.w = np.random.randn(features, 1)
        self.b = np.random.randn(1)
        self.lambda_ = 0

    def pegasos(self, train, test, C, T, f, loss_type='hinge'):
        """
        佩加索斯算法
        Args:
            train: 训练数据
            test: 测试数据
            C: 超参
            T: 训练总轮数
            f: 设定何时画图，建议能整除T
            loss_type: 损失函数类型

        Returns:
            i_list: 保存数据时的训练轮数
            fun_list: 目标函数值
            acc_list: 准确率
        """
        train_x = train['X']  # 4000*1899
        train_y = train['y']  # 4000*1

        test_x = test['Xtest']  # 1000*1899
        test_y = test['ytest']  # 1000*1

        num_train = train_x.shape[0]  # 4000

        lambda_ = 1 / (num_train * C)  # 初始化超参lambda_
        self.lambda_ = lambda_

        i_list = []
        fun_list = []
        acc_list = []

        # 随机训练循环
        for i in range(1, T + 1):
            # TODO 1: 写出下降步长eta_t的计算公式
            # eta_t
            eta_t = 1 / (self.lambda_ * i)

            # 随机选择样本
            choose = np.random.randint(0, num_train)
            x_choose = train_x[[choose]].T  # 1899*1
            y_choose = train_y[choose]

            # 根据不同的loss选择更新w, b
            if loss_type == 'hinge':
                if y_choose * (np.dot(self.w.T, x_choose) + self.b) < 1:
                    self.w = (1 - eta_t * self.lambda_) * self.w + eta_t * y_choose * x_choose
                    self.b = self.b + eta_t * y_choose
                else:
                    self.w = (1 - eta_t * self.lambda_) * self.w


            elif loss_type == 'exp':

                # 计算指数值

                exp_value = -y_choose * (np.dot(self.w.T, x_choose) + self.b)

                if exp_value < 3:  # 判断指数是否过大

                    exp_loss = np.exp(exp_value)

                    self.w = (1 - eta_t * self.lambda_) * self.w + eta_t * exp_loss * y_choose * x_choose

                    self.b = self.b + eta_t * exp_loss * y_choose

                else:

                    # 如果指数过大，跳过当前样本

                    print(f"Skip sample {choose} due to large exponent value.")

                    continue

            elif loss_type == 'log':
                log_loss = 1 / (1 + np.exp(y_choose * (np.dot(self.w.T, x_choose) + self.b)))
                self.w = (1 - eta_t * self.lambda_) * self.w + eta_t * log_loss * y_choose * x_choose
                self.b = self.b + eta_t * log_loss * y_choose

            else:
                raise ValueError('loss_type value error')

            if i % f == 0:
                i_list.append(i)
                fun_now = self.func(train_x, train_y, loss_type)
                fun_list.append(fun_now)
                acc_now = self.acc(test_x, test_y)
                acc_list.append(acc_now)
                print('Epoch: {}, func: {:.4f}, acc: {:.2f}%'.format(i, fun_now, acc_now))

        # 处理f不能整除T时的情况
        if T % f != 0:
            i_list.append(T)
            fun_now = self.func(train_x, train_y, loss_type)
            fun_list.append(fun_now)
            acc_now = self.acc(test_x, test_y)
            acc_list.append(acc_now)
            print('Epoch: {}, func: {:.4f}, acc: {:2f}%'.format(T, fun_now, acc_now))

        return i_list, fun_list, acc_list

    def func(self, data_x, data_y, loss_type='exp'):
        """
        根据当前w，b值计算给定数据上目标函数的平均值
        Args:
            data_x:数据
            data_y:真值
            loss_type:损失函数类型

        Returns:
            返回计算出的平均值
        """
        func = 0
        # TODO 3: 完成该计算函数
        for x, y in zip(data_x, data_y):
            x = x.reshape(-1, 1)
            if loss_type == 'hinge':
                # 计算hinge损失
                loss = max(0, 1 - y * (np.dot(self.w.T, x) + self.b))
                func += loss
            elif loss_type == 'exp':
                # 计算exp损失
                loss = np.exp(-y * (np.dot(self.w.T, x) + self.b))
                func += loss
            elif loss_type == 'log':
                # 计算log损失
                loss = np.log(1 + np.exp(-y * (np.dot(self.w.T, x) + self.b)))
                func += loss
        func /= data_x.shape[0]
        func = func + self.lambda_ * np.dot(self.w.T, self.w) / 2
        return func[0, 0]

    def acc(self, data_x, data_y):
        """
        计算SVM在给定数据上的准确率，
        默认乘以100
        Args:
            data_x: 数据
            data_y: 真值

        Returns:
            计算出的准确率
        """
        num = data_x.shape[0]
        corr = 0
        for x, y in zip(data_x, data_y):
            res = np.dot(x, self.w) + self.b
            if res * y > 0:
                corr += 1
        acc = 100 * corr / num
        return acc
