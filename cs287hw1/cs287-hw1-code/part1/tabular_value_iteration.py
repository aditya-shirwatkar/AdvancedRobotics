import matplotlib.pyplot as plt
from utils.plot import plot_contour, rollout, plot_returns
from utils.utils import upsample
import logger
import numpy as np
import moviepy.editor as mpy
import time


class ValueIteration(object):
    """
    Tabular Value Iteration algorithm.

    -- UTILS VARIABLES FOR RUNNING THE CODE -- (feel free to play with them but we do not ask it for the homework)
        * policy (TabularPolicy):

        * precision (float): tolerance for the final values (determines the amount of iterations)

        * policy_type (str): whether the policy is deterministic or max-ent

        * log_itr (int): number of iterations between logging

        * max_itr (int): maximum number of iterations

        * render (bool): whether to render or not

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

        * self.transitions (np.ndarray): transition matrix of size (S, A, S)

        * self.rewards (np.ndarray): reward matrix of size (S, A, S)

        * self.discount (float): discount factor of the problem

        * self.temperature (float): temperature for the maximum entropy policies


    """
    def __init__(self,
                 env,
                 value_fun,
                 policy,
                 precision=1e-3,
                 log_itr=1,
                 render_itr=2,
                 policy_type='deterministic',
                 max_itr=50,
                 render=True,
                 num_rollouts=20,
                 temperature=1.,
                 ):
        self.env = env
        self.transitions = env.transitions
        self.rewards = env.rewards
        self.value_fun = value_fun
        self.policy = policy
        self.discount = env.discount    
        self.precision = precision
        self.log_itr = log_itr
        assert policy_type in ['deterministic', 'max_ent']
        self.policy_type = policy_type
        self.max_itr = max_itr
        self.render_itr = render_itr
        self.render = render
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.eps = 1e-8

    def train(self):
        next_v = 1e6
        v = self.value_fun.get_values()
        itr = 0
        videos = []
        contours = []
        returns = []
        fig = None

        while not self._stop_condition(itr, next_v, v) and itr < self.max_itr:
            log = itr % self.log_itr == 0
            render = (itr % self.render_itr == 0) and self.render
            if log:
                next_pi = self.get_next_policy()
                self.policy.update(next_pi)
                average_return, video = rollout(self.env, self.policy, render=render,
                                                num_rollouts=self.num_rollouts, iteration=itr)
                if render:
                    contour, fig = plot_contour(self.env, self.value_fun, fig=fig, iteration=itr)
                    contours += [contour] * len(video)
                    videos += video
                returns.append(average_return)
                logger.logkv('Iteration', itr)
                logger.logkv('Average Returns', average_return)
                logger.dumpkvs()
            next_v = self.get_next_values()
            self.value_fun.update(next_v)
            itr += 1

        next_pi = self.get_next_policy()
        self.policy.update(next_pi)
        contour, fig = plot_contour(self.env, self.value_fun, save=True, fig=fig, iteration=itr)
        average_return, video = rollout(self.env, self.policy,
                                        render=self.render, num_rollouts=self.num_rollouts, iteration=itr)
        plot_returns(returns)
        if self.render:
            videos += video
            contours += [contour]
        logger.logkv('Iteration', itr)
        logger.logkv('Average Returns', average_return)

        fps = int(4/getattr(self.env, 'dt', 0.1))
        if contours and contours[0] is not None:
            clip = mpy.ImageSequenceClip(contours, fps=fps)
            clip.write_videofile('%s/contours_progress.mp4' % logger.get_dir())

        if videos:
            clip = mpy.ImageSequenceClip(videos, fps=fps)
            clip.write_videofile('%s/roll_outs.mp4' % logger.get_dir())

        plt.close()

    def get_next_values(self):
        """
        Next values given by the Bellman equation

        :return np.ndarray with the values for each state, shape (num_states,)
        For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                    1) Before computing the exponientated value substract the maximum value per state
                                       over all the actions.
                                    2) Add self.eps to them
        """

        """ INSERT YOUR CODE HERE"""
        if self.policy_type == 'deterministic':
            # print(self.transitions.shape)
            # print(self.rewards.shape)
            # print(self.value_fun.get_values().shape)
            # print(self.discount)
            v = self.value_fun.get_values()
            next_v = np.max((np.sum((self.transitions*(self.rewards + self.discount*v)), axis=2)), axis=1)
            # print(v.shape)
            # raise NotImplementedError
        elif self.policy_type == 'max_ent':
            # raise NotImplementedError
            # v = self.value_fun.get_values()
            # Q = self.rewards + self.discount*v 
            # Q -= np.expand_dims(np.max(Q, axis=1), axis=1)
            # next_v = self.eps*np.log(np.sum( np.sum(np.exp(Q/self.eps), axis=2), axis=1))
            # v = self.value_fun.get_values()
            # beta = 1/self.temperature
            # Q = np.sum(self.rewards + self.discount*v, axis=2)
            # # print(Q)
            # # print(np.expand_dims(np.max(Q, axis=1), axis=1))
            # # Q = Q/np.expand_dims(np.max(Q, axis=1), axis=1)
            # # Q = np.sum(Q, axis=2)
            # Q += self.eps
            # next_v = beta*np.log(np.sum(np.exp(Q/beta), axis=1))
            # v = self.value_fun.get_values()
            # beta = 1/self.temperature
            # Q = self.rewards + self.discount*v
            # # print(Q)
            # # print(np.expand_dims(np.max(Q, axis=1), axis=1))
            # Q = Q - np.expand_dims(np.max(Q, axis=1), axis=1)
            # # Q = np.sum(Q, axis=2)
            # Q += self.eps
            # next_v = np.sum(beta*np.log(np.sum(np.exp(Q/beta), axis=1)), axis=2)
            
            # beta = 1/self.temperature
            # v = self.value_fun.get_values()
            # Q = np.sum(self.rewards, axis=2) + np.expand_dims(self.discount*v, axis=1)
            # Q_sub = np.expand_dims(np.max(Q, axis=1), axis=1)
            # # Q += self.eps
        
            # # Q = Q/(np.expand_dims(np.max(Q, axis=0), axis=0))
            
            # next_v = beta*np.log(np.sum(np.exp(Q/beta - Q_sub), axis=1))
            # next_v += np.max(Q, axis=1)
            # next_v = np.sum(next_v, axis=1)

            # print(v)
            # print(next_v)

            beta = 1/self.temperature
            v = self.value_fun.get_values()
            Q = (self.rewards + self.discount*v)
            Q_sub = np.expand_dims(np.max(Q, axis=1), axis=1)
            next_v = beta*np.log(np.sum(np.exp((Q- Q_sub)/beta), axis=1))
            # next_v += np.max(Q, axis=1)
            next_v = np.max(next_v, axis=1)
            print(next_v)

            """ Your code ends here"""
        else:
            raise NotImplementedError
        return next_v

    def get_next_policy(self):
        """
        Next policy probabilities given by the Bellman equation

        :return np.ndarray with the policy probabilities for each state and actions, shape (num_states, num_actions)
        For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                    1) Before computing the exponientated value substract the maximum value per state
                                       over all the actions.
                                    2) Add self.eps to them
        """

        """INSERT YOUR CODE HERE"""
        if self.policy_type == 'deterministic':
            v = self.value_fun.get_values()
            pi = np.argmax((np.sum((self.transitions*(self.rewards + self.discount*v)), axis=2)), axis=1)

            # raise NotImplementedError
        elif self.policy_type == 'max_ent':
            # raise NotImplementedError
            # v = self.value_fun.get_values()
            # Q = self.rewards + self.discount*v
            # Q -= np.expand_dims(np.max(Q, axis=1), axis=1)
            # z = np.sum( np.exp(np.sum( Q/self.eps, axis=2)) , axis=1)
            # z = np.expand_dims(z, axis=1)
            # pi = (1/z)*np.exp(np.sum( Q/self.eps, axis=2))
            # v = self.value_fun.get_values()
            # beta = 1/self.temperature
            # Q = self.rewards + self.discount*v
            # print(Q)
            # Q = np.sum(Q, axis=2)
            # print(Q)
            # print(np.max(np.max(Q, axis=1), axis=0))
            # # Q = Q/np.expand_dims(np.max(Q, axis=1), axis=1)
            # # print(Q.shape)
            # # Q = np.sum(Q, axis=2)
            # Q += self.eps
            # v = self.value_fun.get_values()
            # beta = 1/self.temperature
            # Q = self.rewards + self.discount*v
            # # print(Q)
            # # print(np.expand_dims(np.max(Q, axis=1), axis=1))
            # Q = Q - np.expand_dims(np.max(Q, axis=1), axis=1)
            # # Q = np.sum(Q, axis=2)
            # Q += self.eps
            # # next_v = np.sum(beta*np.log(np.sum(np.exp(Q/beta), axis=1)), axis=2)
            # z = np.sum(np.exp(Q/beta), axis=1)
            # z = np.expand_dims(z, axis=1) 
            # pi = (1/z)*np.exp(Q/beta)
            # print(pi)
            # p1 = np.sum(pi, axis=2)
            # pi = self.policy.get_probs()
            
            # beta = 1/self.temperature
            # v = self.value_fun.get_values()
            # Q = np.sum(self.rewards, axis=2) + np.expand_dims(self.discount*v, axis=1)
            # print(Q.shape)
            # Q_sub = np.expand_dims(np.max(Q, axis=1), axis=1)
            # # Q += self.eps
        
            # # Q = Q/(np.expand_dims(np.max(Q, axis=0), axis=0))
            # # Q += self.eps
            # # Q = Q/(np.expand_dims(np.max(Q, axis=0), axis=0))
            # print(Q)            
            # z = np.expand_dims(np.sum(np.exp(Q/beta), axis=1), axis=1)
            # pi = (np.exp(Q/beta - Q_sub) + self.eps)/z

            # pi = np.sum(pi, axis=2)

            # print(pi)

            beta = 1/self.temperature
            v = self.value_fun.get_values()
            Q = (self.rewards + self.discount*v)
            Q_sub = np.expand_dims(np.max(Q, axis=1), axis=1)
            # print(Q/beta)
            # print(Q_sub)
            z = np.expand_dims(np.sum(np.exp((Q)/beta), axis=1), axis=1)
            pi = (np.exp(Q/beta - Q_sub) + self.eps)/z
            pi = np.max(pi, axis=2)
            
            """ Your code ends here"""
        else:
            raise NotImplementedError
        return pi

    def _stop_condition(self, itr, next_v, v):
        rmax = np.max(np.abs(self.env.rewards))
        cond1 = np.max(np.abs(next_v - v)) < self.precision/(2 * self.discount/(1 - self.discount))
        cond2 = self.discount ** itr * rmax/(1 - self.discount) < self.precision
        return cond1 or cond2
