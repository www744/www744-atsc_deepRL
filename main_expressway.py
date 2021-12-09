import configparser
from simulator.expressway_env import Expressway
from agent.A2C import A2C, Worker
import tensorflow as tf
import multiprocessing


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)  # #


def write_content(name, date):
    str_content = name + '\n'
    for i in date:
        str_content += ''.join(str(i) + '\n')
    return str_content


def get_reward(config):
    episode_rewards = []
    train_episode_number = config.getint('train_episode_number')
    for i in range(train_episode_number):
        episode_rewards.append(queue.get())
    return episode_rewards


if __name__ == "__main__":
    '''
    加载配置文件
    生成TrafficSimulator类
    '''
    config = configparser.ConfigParser()
    config.read('config/config_expressway.ini')
    env = Expressway(config['ENV_CONFIG'], 7800)
    actor_kwargs = {'hidden_sizes_list': [100, 50], 'learning_rate': 0.00001}
    critic_kwargs = {'hidden_sizes_list': [100, 50], 'learning_rate': 0.00002}
    NUM_WORKERS = multiprocessing.cpu_count()
    with tf.device("/cpu:0"):
        agent = A2C(config['MODEL_CONFIG'], env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
        workers = []
        for i in range(NUM_WORKERS):
            i_name = 'worker{}'.format(i)
            port = 10*i + 8000
            workers.append(Worker(Expressway(config['ENV_CONFIG'], port), agent, name=i_name))

    COORD = tf.train.Coordinator()
    workers_process = []
    queue = multiprocessing.Queue()
    for worker in workers:
        job = worker.work(queue)
        t = multiprocessing.Process(target=job, args=(queue,))
        t.start()
        workers_process.append(t)
    COORD.join(workers_process)
    episode_rewards = get_reward(config['MODEL_CONFIG'])
    print('episode_rewards = ', episode_rewards)
    print('finished\n')
    agent.actor_net.save_weights('./net_weight/expressway/actor_net_weight')
    agent.critic_net.save_weights('./net_weight/expressway/critic_net_weight')
    write_file('./episode_rewards/expressway/episode_rewards.txt', write_content('episode_rewards', episode_rewards))


