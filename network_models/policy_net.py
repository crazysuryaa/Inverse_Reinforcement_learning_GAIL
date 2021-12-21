import tensorflow as tf
import numpy as np

class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.reset()
        # act_space = env.action_space

        # action_space = [
        #     (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        #     (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        #     (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        #     (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        # ]
        number_of_actions = 3

        print(f"in poliucy net observation space: {ob_space.shape}")
        # with sess.as_default():
        #     sess.run(tf.local_variables_initializer())
        #

        with tf.variable_scope(name):
            # self.obs = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape) + [1], name='obs')
            # ob_space = np.expand_dims(ob_space, axis=0)
            self.obs = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                print("input", self.obs.shape)

                layer_1 = tf.layers.conv2d(inputs=self.obs, filters=8, kernel_size=4, strides=2, activation=tf.nn.relu,)
                print(layer_1.shape)

                layer_2 = tf.layers.conv2d(inputs=layer_1, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu,)
                print(layer_2.shape)

                layer_3 = tf.layers.conv2d(inputs=layer_2, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu,)
                print(layer_3.shape)

                layer_4 = tf.layers.conv2d(inputs=layer_3, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, )
                print(layer_4.shape)

                layer_5 = tf.layers.conv2d(inputs=layer_4, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu,)
                print(layer_5.shape)

                layer_6 = tf.layers.conv2d(inputs=layer_5, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu, )
                print(layer_6.shape)

                layer_7 = tf.layers.Flatten()(layer_6)
                print(layer_7.shape)

                layer_8 = tf.layers.dense(inputs=layer_7, units=100, activation=tf.nn.relu)
                print(layer_8.shape)

                self.v_preds  = tf.layers.dense(inputs=layer_8, units=1, activation=None)
                self.act_probs = tf.layers.dense(inputs=layer_8, units=3, activation=None)

                # alpha_head = tf.layers.dense(inputs=layer_8, units=3, activation=tf.softplus())
                # beat_head = tf.layers.dense(inputs=layer_8, units=3, activation=tf.softpus())

                # alpha, beta = self.net(s[index])[0]
                # dist = Beta(alpha, beta)
                # a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                # ratio = torch.exp(a_logp - old_a_logp[index])


                # # self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
                # # self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
                # self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
                # self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())

                # self.act_probs = tf.layers.dense(inputs=layer_3, units=number_of_actions, activation=tf.nn.softmax)

                # layer_3 = tf.layers.dense(inputs=layer_7, units=act_space.n, activation=tf.tanh)



            # with tf.variable_scope('value_net'):
            #     layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
            #     layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
            #     self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            # self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            # self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
            #
            # self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        # obs = np.expand_dims(obs, axis=3)
        return tf.get_default_session().run([self.act_probs, self.v_preds], feed_dict={self.obs: obs})

        # if stochastic:
        #     return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        # else:
        #     return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
