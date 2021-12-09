import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

from tensorflow import keras

tf.random.set_seed(0)
np.random.seed(0)


class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

    def plot_episode_rewards(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.canvas.draw()
        plt.savefig('./episode_rewards/episode_rewards.png')
        plt.show()


class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99, actor_net_weight=None,
                 critic_net_weight=None):
        self.action_n = env.action_space
        self.gamma = gamma
        self.discount = 1.
        self.actor_net_weight = actor_net_weight
        self.critic_net_weight = critic_net_weight
        self.actor_net = self.build_actor_network(output_size=self.action_n,
                output_activation=tf.nn.softmax,
                loss=tf.losses.categorical_crossentropy,
                **actor_kwargs)
        self.critic_net = self.build_critic_network(output_size=self.action_n,
                                             **critic_kwargs)

    def build_actor_network(self, hidden_sizes, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        if self.actor_net_weight :
            model.load_weights(self.actor_net_weight)
        return model

    def build_critic_network(self, hidden_sizes, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        ob = np.array(observation)
        probs = self.actor_net.predict(ob[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation,
              done, next_action=None):
        observation = np.array(observation)
        next_observation = np.array(next_observation)
        # 训练执行者网络
        x = observation[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor,
                                                        1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
            grad_tensors, self.actor_net.variables))

        # 训练评论者网络
        u[0, action] = reward
        if not done:
            q = self.critic_net.predict(
                next_observation[np.newaxis])[0, next_action]
            u[0, action] += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


class AdvantageActorCriticAgent(QActorCriticAgent):
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99, actor_net_weight=None,
                 critic_net_weight=None):
        self.action_n = env.action_space
        self.gamma = gamma
        self.discount = 1.
        self.actor_net_weight = actor_net_weight
        self.critic_net_weight = critic_net_weight
        self.actor_net = self.build_actor_network(output_size=self.action_n,
                output_activation=tf.nn.softmax,
                loss=tf.losses.categorical_crossentropy,
                **actor_kwargs)
        self.critic_net = self.build_critic_network(output_size=1,
                                             **critic_kwargs)

    def learn(self, observation, action, reward, next_observation, done, next_action=None):
        observation = np.array(observation)
        next_observation = np.array(next_observation)
        x = observation[np.newaxis]
        u = reward + (1. - done) * self.gamma * \
            self.critic_net.predict(next_observation[np.newaxis])
        td_error = u - self.critic_net.predict(x)

        # 训练执行者网络
        x_tensor = tf.convert_to_tensor(observation[np.newaxis],
                                        dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor,
                                                        1e-6, 1.))
            loss_tensor = -self.discount * td_error * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
            grad_tensors, self.actor_net.variables))  # 更新执行者网络

        # 训练评论者网络
        self.critic_net.fit(x, u, verbose=0)  # 更新评论者网络

        if done:
            self.discount = 1.  # 为下一回合初始化累积折扣
        else:
            self.discount *= self.gamma  # 进一步累积折扣







