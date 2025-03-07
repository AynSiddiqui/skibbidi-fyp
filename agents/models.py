"""
A2C, IA2C, MA2C models
@author: Tianshu Chu
"""

import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from agents.policies import COMAPolicy

class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        if self.name == 'ma2c':
            n_fp = model_config.getint('num_fp')
            # policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
            #                         n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
            policy = FPLstmACPolicyEdited(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                    n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        else:
            policy = LstmACPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, os.path.join(model_dir, save_file))
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, done)


class IA2C(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma))

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        #Edit: Add advs_ls to check advantage
        advs_ls = []
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            advs_ls.append(Advs)
            if i == 0:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                           summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)
        return advs_ls

    def forward(self, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(self.sess, obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            return out1, out2

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], values[i], done)


class MA2C(IA2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls, total_step,
                 model_config, seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())


class IQL(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0, model_type='dqn'):
        self.name = 'iql'
        self.model_type = model_type
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s, n_a, n_w, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.cur_step = 0
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, model_config, agent_name=None):
        if self.model_type == 'dqn':
            n_h = model_config.getint('num_h')
            n_fc = model_config.getint('num_fc')
            policy = DeepQPolicy(n_s - n_w, n_a, n_w, self.n_step, n_fc0=n_fc, n_fc=n_h,
                                 name=agent_name)
        else:
            policy = LRQPolicy(n_s, n_a, self.n_step, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))

    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                if i == 0:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                               summary_writer=summary_writer,
                                               global_step=global_step + k)
                else:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr)

    def forward(self, obs, mode='act', stochastic=False):
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        action = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if (mode == 'explore') and (np.random.random() < eps):
                action.append(np.random.randint(self.n_a_ls[i]))
            else:
                if not stochastic:
                    action.append(np.argmax(qs))
                else:
                    qs = qs / np.sum(qs)
                    action.append(np.random.choice(np.arange(len(qs)), p=qs))
            qs_ls.append(qs)
        return action, qs_ls

    def reset(self):
        # do nothing
        return

    def add_transition(self, obs, actions, rewards, next_obs, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], next_obs[i], done)
class DQN:
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0):
        self.name = 'dqn'
        self.n_agent = len(n_s_ls)
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            self.policy_ls.append(self._init_policy(n_s, n_a, n_w, model_config, agent_name="{}a".format(i)))
            
        self.saver = tf.train.Saver(max_to_keep=5)
        
        if total_step:
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        
        self.sess.run(tf.global_variables_initializer())
        
    def _init_policy(self, n_s, n_a, n_w, model_config, agent_name=None):
        n_h = model_config.getint('num_h')
        n_fc = model_config.getint('num_fc')
        return DeepQPolicy(n_s - n_w, n_a, n_w, self.n_step, n_fc0=n_fc, n_fc=n_h, name=agent_name)
        
    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
            
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio_str = model_config.get('epsilon_ratio')
            if eps_ratio_str is None:
                eps_ratio = 1.0
            else:
                try:
                    eps_ratio = float(eps_ratio_str)
                except Exception:
                    eps_ratio = 1.0
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio, decay=eps_decay)
    
    def _init_train(self, model_config):
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size_str = model_config.get('buffer_size')
        if buffer_size_str is None:
            buffer_size = 1000.0
        else:
            try:
                buffer_size = float(buffer_size_str)
            except Exception:
                buffer_size = 1000.0
        
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))
    
    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                           summary_writer=summary_writer,
                                           global_step=global_step + k if global_step else None)
    
    def forward(self, obs, mode='act', stochastic=False):
        eps = self.eps_scheduler.get(1) if mode == 'explore' else 0.0
        actions = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if mode == 'explore' and np.random.random() < eps:
                actions.append(np.random.randint(self.n_a_ls[i]))
            else:
                if not stochastic:
                    actions.append(np.argmax(qs))
                else:
                    qs = qs / np.sum(qs)
                    actions.append(np.random.choice(np.arange(len(qs)), p=qs))
            qs_ls.append(qs)
        return actions, qs_ls
    
    def reset(self):
        pass
    
    def add_transition(self, obs, actions, rewards, next_obs, done):
        if self.reward_norm:
            rewards /= self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
    
    def save(self, model_dir, global_step):
        self.saver.save(self.sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    
    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{}'.format(int(checkpoint))
        if save_file:
            self.saver.restore(self.sess, os.path.join(model_dir, save_file))
            logging.info('Checkpoint loaded: {}'.format(save_file))
            return True
        logging.error('Cannot find old checkpoint for {}'.format(model_dir))
        return False

class COMA:
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0):
        self.name = 'coma'
        self.n_agent = len(n_s_ls)
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        
        # Create one COMAPolicy per agent.
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            self.policy_ls.append(COMAPolicy(n_s, n_a, n_w, self.n_step, model_config, name="coma_{}".format(i)))
        
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())
    
    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
    
    def _init_train(self, model_config):
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.get('buffer_size')
        try:
            buffer_size = float(buffer_size)
        except Exception:
            buffer_size = 1000.0
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))
    
    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                self.policy_ls[i].backward(self.sess, obs, acts, dones, rs, cur_lr,
                                           summary_writer=summary_writer,
                                           global_step=global_step + k if global_step else None)
    
    def forward(self, obs, mode='act', stochastic=False):
        # For COMA, we assume each agent requires both its local observation and the global state.
        actions = []
        qs_ls = []
        for i in range(self.n_agent):
            # Here, obs[i] should be a tuple: (local_obs, global_state)
            act, qs = self.policy_ls[i].forward(self.sess, obs[i], mode, stochastic)
            actions.append(act)
            qs_ls.append(qs)
        return actions, qs_ls
    
    def reset(self):
        for policy in self.policy_ls:
            policy.reset()
    
    def add_transition(self, obs, actions, rewards, next_obs, done):
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
    
    def save(self, model_dir, global_step):
        self.saver.save(self.sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
    
    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{}'.format(int(checkpoint))
        if save_file:
            self.saver.restore(self.sess, os.path.join(model_dir, save_file))
            logging.info('Checkpoint loaded: {}'.format(save_file))
            return True
        logging.error('Cannot find old checkpoint for {}'.format(model_dir))
        return False