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
        MainWindow.resize(600, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.set_matplotlib()

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 50, 100, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.start_sim)


        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "经济引擎模拟"))
        self.pushButton.setText(_translate("MainWindow", "开始模拟"))


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
        self.obs = self.env.reset()

    # 嵌入matplotlib方法
    def set_matplotlib(self):
        # 创建画布
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)

        # 把画布放进widget组件,设定位置
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.canvas)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setLayout(self.vlayout)
        # self.widget.setGeometry(QtCore.QRect(240, 20, 600, 400))
        self.widget.setObjectName("matplotlib")
        self.widget.setGeometry(QtCore.QRect(0, 0, 600,900))

        # 初始化matplotlib显示区域

        self.load_env()
        self.set_drawing()

    def set_drawing(self):
        max_health = {"Wood": 1, "Stone": 1, "House": 1}


        self.ax.cla()
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



        self.ax.imshow(tmp, vmax=1.0, aspect="auto")
        bbox = self.ax.get_window_extent()
        for i in range(n_agents):
            r, c = locs[cmap_order[i]]
            col = np.array(cmap(i)[:3])
            self.ax.plot(c, r, "o", markersize=bbox.height * 20 / 550, color="w")
            self.ax.plot(c, r, "*", markersize=bbox.height * 15 / 550, color=col)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def start_sim(self):

        self.testTimer = QtCore.QTimer()
        self.testTimer.timeout.connect(self.plotfig)  # 调用绘图方法
        self.testTimer.start(10)


# 绘图方法
    def plotfig(self):
        for t in range(self.env.episode_length):
            actions = self.sample_random_actions(self.env, self.obs)
            obs, rew, done, info = self.env.step(actions)
            if ((t + 1) % 2) == 0:
                self.set_drawing()
        self.testTimer.stop()

    def sample_random_action(self,agent, mask):
        if agent.multi_action_mode:
            split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
            return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]
        else:
            return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())

    def sample_random_actions(self,env, obs):
        actions = {a_idx: self.sample_random_action(self.env.get_agent(a_idx), a_obs['action_mask'])for a_idx, a_obs in obs.items()}
        return actions

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(w)
    w.show()
    app.exec_()