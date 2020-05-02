import os
import gym
import argparse
from collections import deque
from ma_gym.wrappers import Monitor
from multiprocessing import Process, Queue

def add_player(player, mode, q1, q2):
    player.start(mode, q1, q2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='PongDuel-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=500000,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    reward_list = deque(maxlen=100)
    env = gym.make(args.env)
    #env = Monitor(env, directory='testings/' + args.env, force=True)

    left_q1 = Queue()
    left_q2 = Queue()
    left_q3 = Queue()
    right_q1 = Queue()
    right_q2 = Queue()

    # studnetID 는 각 학생들 학번으로 변경 후 평가합니다..
    studentID1 = "sample"
    studentID2 = "sample"
    left_player = __import__(studentID1)
    right_player = __import__(studentID2)
    left_p = Process(target=add_player, args=(left_player, "left", left_q1, left_q2))
    right_p = Process(target=add_player, args=(right_player, "right", right_q1, right_q2))
    left_p.start()
    right_p.start()

    l_cnt = 0
    r_cnt = 0
    for ep_i in range(args.episodes): 
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()

        while not all(done_n): 

            left_q1.put( obs_n )
            right_q1.put( obs_n )
            l_action = left_q2.get()
            r_action = right_q2.get()
            obs_n, reward_n, done_n, info = env.step([l_action, r_action])
            
            print(obs_n)
            print(aaaaa)

            #action_n = env.action_space.sample()
            #obs_n, reward_n, done_n, info = env.step(action_n)

            l_reward = reward_n[0]
            r_reward = reward_n[1]
            l_cnt += l_reward
            r_cnt += r_reward

        #print('Episode #{} left: {} right: {} '.format(ep_i, l_cnt, r_cnt))

    left_q1.put( None )
    right_q1.put( None )
    left_p.join()
    right_p.join()
    env.close()

    print("Final records:", l_cnt, "vs.", r_cnt)