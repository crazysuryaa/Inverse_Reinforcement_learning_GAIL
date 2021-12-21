#!/usr/bin/python3
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
from expert_trajectories import Agent
from environment import Env

# Used for cloud/Agave running
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

trainedWeights = 'param/ppo_net_params.pkl'

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    return parser.parse_args()


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def main(args):

    env = Env()
    expert = Agent(trainedWeights)
    expert.load_param()

    render = True

    logdir = os.path.join("logs", str(time.time()))
    savedir = os.path.join(logdir, "saved_model")

    print(f"Logdir: {logdir}\n savedir: {savedir}")

    obs = env.reset()
    reward = 0  # do NOT use rewards to update policy
    model_save_index = 10

    exp_rwd_iter = []
    expert_obs = []

    expert_obs_array = np.empty((obs.shape))
    expert_acts = []
    state_space = np.concatenate((obs, obs), axis=3)
    expert_states_array = np.empty((state_space.shape))

    # Getting Agent states and actions
    episode_rewards = []
    expert_iterations = 1
    expert_actions_array = np.empty((1, 3))

    # Getting Expert states and actions
    for iteration in range(expert_iterations):

        ep_obs = []
        ep_rwds = []

        t = 0
        done = False
        #Input shape: (1, 96, 96, 4)
        ob = env.reset()
        steps = 0

        # print(f"Input shape: {ob.shape}")

        while not done:

            state_space = obs.copy()

            # Expert action: [0.99428356 0.5814981  0.0103017](3, )
            act = expert.act(ob)
            # print("Expert action:", act, act.shape)

            ep_obs.append(ob)
            expert_obs.append(ob)

            action = np.array(act)
            expert_acts.append(action)
            expert_obs_array = np.vstack((expert_obs_array, obs))
            expert_actions_array = np.vstack((expert_actions_array, action))

            if render:
                env.render()

            ob, rwd, done, info = env.step(act * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            state_space = np.concatenate((state_space, obs), axis=3)
            expert_states_array = np.vstack((expert_states_array, state_space))

            ep_rwds.append(rwd)

            t += 1
            steps += 1

            # if t >= 1:
            #     break

        if done:
            exp_rwd_iter.append(np.sum(ep_rwds))

        print(f"Expert Episode:{iteration} - Expert Reward: {np.sum(ep_rwds)}")


    # exit(0)
    # Initialising Policy and Descriminator networks
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(env)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        writer = tf.summary.FileWriter(logdir, sess.graph)

        obs = env.reset()

        for iteration in range(args.iteration):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            observ_array = np.empty((obs.shape))

            state_space = np.concatenate((obs, obs), axis=3)
            agents_states_array = np.empty((state_space.shape))

            actions_array = np.empty((1,3))

            while True:
                run_policy_steps += 1

                # Get Agent action for the given state
                agent_act, v_pred = Policy.act(obs=obs, stochastic=True)

                observations.append(obs)
                actions.append(agent_act)
                v_preds.append(v_pred)
                rewards.append(reward)

                observ_array = np.vstack((observ_array, obs))
                actions_array = np.vstack((actions_array, agent_act))
                agent_act = agent_act[0]

                next_obs, reward, done, info = env.step(agent_act * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

                state_space = np.concatenate((obs, next_obs), axis=3)
                agents_states_array = np.vstack((agents_states_array, state_space))

                if render:
                    env.render()

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            episode_total_rewards = sum(rewards)
            episode_rewards.append(episode_total_rewards)

            print(f"Agent Episode: {iteration} - Total rewards: {episode_total_rewards}")

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=episode_total_rewards)])
                               , iteration)

            if iteration % model_save_index == 0:
                save_path = saver.save(sess, "saved_model/model.ckpt")
                print("Model saved in path: %s" % save_path)

            expert_acts = np.array(expert_acts).astype(dtype=np.int32)

            # train discriminator
            for i in range(5):
                # D.train(expert_s=expert_states_array,
                #         agent_s=agents_states_array,
                #         # exper_a=
                #         )

                D.train(expert_s=expert_obs_array,
                        agent_s=observ_array,
                        expert_a=expert_actions_array,
                        agent_a=actions_array
                        )

            d_rewards = D.get_rewards(agent_s=observ_array,agent_a=actions_array)
            print(f" Rewards from Descriminator:{d_rewards}")

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)

            # gaes = (gaes - gaes.mean()) / gaes.std()
            # gaes = gaes.reshape(gaes.shape[0], gaes.shape[1] * gaes.shape[2])

            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            v_preds_next = np.expand_dims(v_preds_next, axis=1)

            print(f"observations: {observ_array.shape}, Actions: {actions_array.shape}, gaes: {gaes.shape}, ")
            print(f"rewards: {d_rewards.shape}, v_next: {v_preds_next.shape}")

            policy_epochs = 50

            PPO.assign_policy_parameters()
            min_index = min([len(observations), len(actions), len(gaes), len(d_rewards), len(v_preds_next)])

            # Training Policy (Generator)
            for epoch in range(policy_epochs):

                rand_index = np.random.randint(0, min_index)

                PPO.train(obs=observations[rand_index],
                          actions=actions[rand_index],
                          gaes=gaes[rand_index],
                          rewards=d_rewards[rand_index],
                          v_preds_next=v_preds_next[rand_index])

                summary = PPO.get_summary(obs=observations[rand_index],
                          actions=actions[rand_index],
                          gaes=gaes[rand_index],
                          rewards=d_rewards[rand_index],
                          v_preds_next=v_preds_next[rand_index])

            writer.add_summary(summary, iteration)

        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
