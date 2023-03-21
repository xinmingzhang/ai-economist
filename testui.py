from PyQt5 import QtWidgets
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import  QApplication,QVBoxLayout, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ai_economist.foundation import landmarks, resources

import os, signal, sys, time
from ai_economist import foundation
import numpy as np

import matplotlib.pyplot as plt
import myplotting as plotting

plt.rcParams["font.sans-serif"] = ["Simhei"]  # 设置默认字体
plt.rcParams["axes.unicode_minus"] = False  # 坐标轴正确显示正负号


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Pyqt5时序打印")
        MainWindow.resize(900, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.set_matplotlib()

        # 速度标签
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 120, 21))
        self.label.setObjectName("label")
        # 速度输入框
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(20, 50, 200, 21))
        self.lineEdit.setObjectName("lineEdit")

        # 线条颜色标签
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 100, 101, 21))
        self.label_2.setObjectName("label_2")
        # Max点标签
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 300, 50, 20))
        self.label_3.setObjectName("label_3")
        # Min点标签
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 350, 50, 20))
        self.label_4.setObjectName("label_4")
        # 选择文件标签
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(70, 300, 180, 21))
        self.label_5.setObjectName("label_5")
        # 显示文件地址标签
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(70, 350, 180, 21))
        self.label_6.setObjectName("label_6")

        # 默认线条红色
        self.line_color = 'r'
        # 颜色单选框三个
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(20, 130, 86, 21))
        self.radioButton.setObjectName("radioButton")
        self.radioButton.toggled.connect(self.get_color1)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 160, 86, 21))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_2.toggled.connect(self.get_color2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 190, 86, 21))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.toggled.connect(self.get_color3)

        # 开始绘图按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 250, 101, 31))
        self.pushButton.setObjectName("pushButton")
        # 点击开始绘图执行定时器方法（需要自己编写）
        self.pushButton.clicked.connect(self.start_dingshiqi)
        # 选择文件按钮
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(20, 420, 101, 31))
        self.pushButton_1.setObjectName("pushButton_1")
        # 点击按钮执行选择文件方法
        self.pushButton_1.clicked.connect(self.choice_file)
        # 显示文件路径的标签
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(130, 420, 500, 31))
        self.label_7.setObjectName("label_7")
        self.file_dir = None

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.i = 0
        self.t_list = []
        self.y_list = []

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "经济引擎模拟"))
        self.label.setText(_translate("MainWindow", "绘图速度（秒）："))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "请输入速度，默认为1"))

        self.label_2.setText(_translate("MainWindow", "线条颜色："))
        self.label_3.setText(_translate("MainWindow", "Max点："))
        self.label_4.setText(_translate("MainWindow", "Min点："))

        self.radioButton.setText(_translate("MainWindow", "red"))
        self.radioButton_2.setText(_translate("MainWindow", "blue"))
        self.radioButton_3.setText(_translate("MainWindow", "green"))

        self.pushButton.setText(_translate("MainWindow", "开始绘图"))
        self.pushButton_1.setText(_translate("MainWindow", "选择文件"))

    def load_env(self):
        env_config = {
            'scenario_name': 'layout_from_file/simple_wood_and_stone',
            'components': [
                ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
                ('ContinuousDoubleAuction', {'max_num_orders': 5}),
                ('Gather', {}),
            ],
            'env_layout_file': 'my_map_sperate.txt',
            'starting_agent_coin': 10,
            'fixed_four_skill_and_loc': False,
            'n_agents': 3,
            'world_size': [15, 10],
            'episode_length': 1000,
            'multi_action_mode_agents': False,
            'multi_action_mode_planner': True,
            'flatten_observations': False,
            'flatten_masks': True,
        }

        self.env = foundation.make_env_instance(**env_config)

    # 嵌入matplotlib方法
    def set_matplotlib(self):
        # 创建画布
        self.fig = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.fig)

        # 把画布放进widget组件,设定位置
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.canvas)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setLayout(self.vlayout)
        # self.widget.setGeometry(QtCore.QRect(240, 20, 600, 400))
        self.widget.setObjectName("matplotlib")

        # 初始化matplotlib显示区域

        self.load_env()

        max_health = {"Wood": 1, "Stone": 1, "House": 1}
        self.env.reset()


        scenario_entities = [k for k in self.env.world.maps.keys() if "source" not in k.lower()]

        # plotting.plot_env_state(env)
        locs = [agent.loc for agent in self.env.world.agents]


        from ai_economist.foundation import landmarks, resources
        maps = self.env.world.maps
        world_size = np.array(maps.get("Wood")).shape
        tmp = np.zeros((3, world_size[0], world_size[1]))

        for entity in scenario_entities:
            if entity == "House":
                continue
            elif resources.has(entity):
                if resources.get(entity).collectible:
                    map_ = (resources.get(entity).color[:, None, None] * np.array(maps.get(entity))[None])
                    map_ /= max_health[entity]
                    tmp += map_
            elif landmarks.has(entity):
                map_ = (landmarks.get(entity).color[:, None, None] * np.array(maps.get(entity))[None])
                print(map_)
                tmp += map_
            else:
                continue

        n_agents = len(locs)
        cmap = plt.get_cmap("jet", n_agents)
        cmap_order = list(range(n_agents))

        if isinstance(maps, dict):
            house_idx = np.array(maps.get("House")["owner"])
            house_health = np.array(maps.get("House")["health"])
        else:
            house_idx = maps.get("House", owner=True)
            house_health = maps.get("House")
        for i in range(n_agents):
            houses = house_health * (house_idx == cmap_order[i])
            agent = np.zeros_like(houses)
            agent += houses
            col = np.array(cmap(i)[:3])
            map_ = col[:, None, None] * agent[None]
            tmp += map_

        tmp *= 0.7
        tmp += 0.3

        tmp = np.transpose(tmp, [1, 2, 0])
        tmp = np.minimum(tmp, 1.0)


        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(tmp, vmax=1.0, aspect="auto")
        bbox = self.ax.get_window_extent()
        for i in range(n_agents):
            r, c = locs[cmap_order[i]]
            col = np.array(cmap(i)[:3])
            self.ax.plot(c, r, "o", markersize=bbox.height * 20 / 550, color="w")
            self.ax.plot(c, r, "*", markersize=bbox.height * 15 / 550, color=col)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

# 选择文件方法
    def choice_file(self):
        # 打开文件，获取文件地址
        self.filedir, self.filetype = QFileDialog.getOpenFileName(None, "选取文件", "", "*.*")
        self.label_7.setText('文件地址：%s' % self.filedir)
        self.file_dir = self.filedir  # 将文件地址展示在界面中
        # 执行读文件方法
        self.read_file()

 # 读文件方法
    def read_file(self):
        # 读入文件
        self.data = pd.read_csv(self.file_dir, header=None, encoding='utf-8')
        # x轴需要绘制的所有数据
        self.t = self.data.index
        # y轴需要绘制的所有数据
        self.y = self.data.values
        self.t = np.array([x for x in range(1000)])
        self.y = np.array([x+1 for x in range(1000)])

    # 颜色单选框方法
    def get_color1(self):
        self.line_color = 'r'

    def get_color2(self):
        self.line_color = 'b'

    def get_color3(self):
        self.line_color = 'g'

    # # 找最大值方法
    # def search_max(self):
    #     self.ymax_data = max(self.data.values)
    #     for i, j in enumerate(self.data.values):
    #         if j == self.ymax_data:
    #             self.xmax_data = i
    #     self.label_5.setText('(%d，%f)' % (self.xmax_data, self.ymax_data))
    #
    # # 找最小值方法
    # def search_min(self):
    #     self.ymin_data = min(self.data.values)
    #     for i, j in enumerate(self.data.values):
    #         if j == self.ymin_data:
    #             self.xmin_data = i
    #     self.label_6.setText('(%d，%f)' % (self.xmin_data, self.ymin_data))

# 定时器方法
    def start_dingshiqi(self):
        self.ax.cla()
        self.ax.set_xlim(0, len(self.t) + 1)
        self.ax.set_ylim(min(self.y - 1), max(self.y) + 1)
        self.ax.set_yticks(np.arange(min(self.y) - 1, max(self.y) + 1, 5))
        self.ax.set_xticks(np.arange(0, len(self.t) + 1, 500))
        self.ax.set_title('正在绘图')
        self.ax.set_xlabel('序号')
        self.ax.set_ylabel('数值')

        self.testTimer = QtCore.QTimer()
        self.testTimer.timeout.connect(self.plotfig)  # 调用绘图方法
        self.testTimer.start(10)


# 绘图方法
    def plotfig(self):
        self.ax.autoscale_view()
        # 绘图
        self.ax.plot(self.t_list, self.y_list, c=self.line_color, linewidth=1)
        self.fig.canvas.draw()  # 画布重绘，self.figs.canvas
        self.fig.canvas.flush_events()  # 画布刷新 self.figs.canvas
        self.t_list.append(self.t[self.i])  # 更新数据
        self.y_list.append(self.y[self.t[self.i]])  # 每次给原来数据加入新数据
        self.i += 10
        if self.i >= len(self.t):
            self.testTimer.stop()

if __name__ == '__main__':


    app = QApplication(sys.argv)

    w = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(w)
    w.show()
    app.exec_()