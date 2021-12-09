"""
A2C algorithms
@author: wdz
"""

import numpy as np
import configparser
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = configparser.ConfigParser()
config.read('./config/config_expressway.ini')
ENTROY_BETA = 0.01
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
GLOBAL_STEPS = 0
MAX_GLOBAL_EP = config.getint('MODEL_CONFIG', 'train_episode_number')
COORD = tf.train.Coordinator()

class A2C:
    def __init__(self, config, env, actor_kwargs, critic_kwargs, evaluate=False, actor_net_weight=None,
                 critic_net_weight=None):
        self.name = 'A2C'
        self.action_n = env.action_space
        self.state_n = env.state_space
        self.gamma = config.getfloat('gamma')
        self.discount = config.getfloat('discount')
        self.actor_net_weight = actor_net_weight
        self.critic_net_weight = critic_net_weight
        self.evaluate = evaluate
        self.actor_net = self.build_actor_network(output_size=self.action_n,
                input_size=self.state_n, output_activation=tf.nn.softmax,
                **actor_kwargs)
        self.critic_net = self.build_critic_network(output_size=1,
                input_size= self.state_n, **critic_kwargs)
        self.critic_net_copy = self.critic_net

    def build_actor_network(self, hidden_sizes_list, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      learning_rate=0.001):
        model = tf.keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes_list):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(tf.keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        model.add(tf.keras.layers.Dense(units=output_size, activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer)
        if self.actor_net_weight :
            model.load_weights(self.actor_net_weight)
        model.summary()
        return model

    def build_critic_network(self, hidden_sizes_list, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = tf.keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes_list):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(tf.keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        model.add(tf.keras.layers.Dense(units=output_size, activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        if self.critic_net_weight:
            model.load_weights(self.critic_net_weight)
        model.summary()
        return model

    def decide(self, observation):
        ob = np.array(observation)
        probs = self.actor_net.predict(ob[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, batch_n, done):
        y = reward.reshape((len(reward), 1)) + (1. - done) * self.gamma * \
            self.critic_net_copy.predict(next_observation)
        # 更新评论者网络fit(x, y, batch_size=None, epochs=1);
        # x:state,y:r + gamma * predict(next_state)
        self.critic_net.fit(observation, y, verbose=0)
        if batch_n % 10 == 0:
            self.critic_net_copy = self.critic_net
        # 训练执行者网络
        advantage = reward.reshape((len(reward), 1)) + self.critic_net.predict(next_observation) - \
                    self.critic_net.predict(observation)
        # 计算优势函数 r + gamma * v(next_state) - v(state)
        x_tensor = tf.convert_to_tensor(observation[np.newaxis], dtype=tf.float32)
        ty = []
        for i in range(len(action)):
            ty.append([0, i, action[i]])
        # self.update(x_tensor, ty, advantage)

    # @tf.function
    # def update(self, x_tensor, ty, advantage):
        with tf.GradientTape() as tape:
            pi_tensor = tf.gather_nd(self.actor_net(x_tensor), ty)
            # 当前动作的概率
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            # 取当前动作的概率，并求log
            loss_tensor = -tf.math.reduce_sum(self.discount * advantage * logpi_tensor)
            # print('loss_tensor = \n', loss_tensor)
            # 计算损失函数
            # 需要增加entropy代码
            entropy_loss = loss_tensor + ENTROY_BETA * tf.math.reduce_sum(pi_tensor * tf.math.log(pi_tensor + 1e-5))
        grad_tensors = tape.gradient(entropy_loss, self.actor_net.variables)
        # 求损失函数梯度tape.gradient(损失函数，变量名)
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))
        # 更新执行者网络


class Worker:
    def __init__(self, env, agent, name):
        self.env = env
        self.name = name
        self.AC = agent
        self.step_counter = 0
        self.total_reward = 0
        self.gamma = 0.99
        self.actor_net = self.AC.actor_net
        self.critic_net = self.AC.critic_net

    def work(self, queue):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            episode_reward = 0
            self.env.terminate()
            observation = self.env.reset()
            step = 0
            batch = 1
            while True:
                action = self.AC.decide(observation)
                actioncode = [0, 0, 0]
                if action == 0:
                    actioncode = [0, 0, 0]
                if action == 1:
                    actioncode = [0, 0, 1]
                if action == 2:
                    actioncode = [0, 1, 0]
                if action == 3:
                    actioncode = [0, 1, 1]
                if action == 4:
                    actioncode = [1, 0, 0]
                if action == 5:
                    actioncode = [1, 0, 1]
                if action == 6:
                    actioncode = [1, 1, 0]
                if action == 7:
                    actioncode = [1, 1, 1]
                next_observation, reward, done, _ = self.env.step(actioncode)
                observation = np.array(observation)
                next_observation = np.array(next_observation)
                buffer_s.append(observation)
                buffer_s_.append(next_observation)
                buffer_a.append(action)
                buffer_r.append(reward)
                episode_reward += reward
                if len(buffer_a) % 20 == 0:
                    batch += 1
                    buffer_s = np.vstack(buffer_s)
                    buffer_s_ = np.vstack(buffer_s_)
                    buffer_a = np.array(buffer_a)
                    buffer_r = np.array(buffer_r)
                    self.AC.learn(buffer_s, buffer_a, buffer_r, buffer_s_, batch, done)
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
                step += 1
                observation = next_observation
                if done:
                    GLOBAL_RUNNING_R.append(episode_reward)
                    # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    #     GLOBAL_RUNNING_R.append(episode_reward)
                    # else:
                    #     GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * episode_reward)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP, "| Ep_r: %i" % GLOBAL_RUNNING_R[-1], )
                    GLOBAL_EP += 1
                    queue.put(episode_reward)
                    break


