import mnist_loader
import network
import os

if __name__ == '__main__':
    # 改变工作路径至当前文件目录
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # 一些可以调整的超参数
    epochs = 20
    mini_batch_size = 10
    eta = 0.1
    net_sizes = [784, 192, 30, 10]
    output_pic = 'epochs' + str(epochs) + '_result.jpg'

    #数据集
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #网络
    net = network.Network(net_sizes, cost=network.CrossEntropyCost)

    #输出
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data, epochs, mini_batch_size, eta, evaluation_data=test_data,
                  monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
                  monitor_training_cost=True, monitor_training_accuracy=True)

    network.plot_result(epochs, test_cost, test_accuracy, training_cost, training_accuracy, output_pic)
