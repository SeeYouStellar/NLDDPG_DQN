import hyperparameter as hp
from agent import DDPG
from env import Env
from replaymemory import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losss, name):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Step(*10)')
    plt.ylabel(name)
    plt.plot(losss)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':

    env = Env()

    state_dim = 1 + 3 * hp.M
    action_dim = 2
    agent = DDPG(state_dim, action_dim)
    RB = ReplayBuffer(hp.MEMORY_CAPACITY)
    episode = 0
    step = 0
    mean_actor_losses = []
    mean_critic_losses = []
    actor_losses = []
    critic_losses = []
    rewards = []
    mean_rewards = []
    td_errors = []
    mean_td_errors = []
    step = 0
    while episode < hp.MAX_EPISODES:
        done = 0
        s = env.reset()
        episode += 1
        for i in range(hp.MAX_STEPS):
            a = agent.choose_action(s)
            # a = np.clip(np.random.normal((a + 1) / 2, 0.01), -1, 1)
            s_, r, done, redo = env.step(a)
            # with open("ar", 'a') as file:
            #     file.write("action is {}, reward is {}\n".format(a, r))
            if done:
                break
            if redo:
                continue
            with open('log1', 'a') as file:
                file.write('ve:{} \t ratio:{} \t r:{} \n'.format(a[0], a[1], r))
            RB.push(s, a, r, done, s_)
            step += 1
            s = s_
            rewards.append(r)
            if step > hp.MEMORY_CAPACITY:
                bs, ba, br, bd, bs_ = RB.sample(n=hp.BATCH_SIZE)
                actor_loss, critic_loss= agent.learn(bs, ba, br, bd, bs_)
                # td_errors.append(td_error)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                # if step % 50 == 0:
                #     agent.actor_target.load_state_dict(agent.actor.state_dict())
                #     agent.critic_target.load_state_dict(agent.critic.state_dict())
                # if step % hp.plot_frequency == 0:
    plot_loss(actor_losses, 'actor_loss')
    plot_loss(critic_losses, 'critic_loss')
    plot_loss(rewards, 'reward')
                    # plot_loss(mean_td_errors, 'td_error')
