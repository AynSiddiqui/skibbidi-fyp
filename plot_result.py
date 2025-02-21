from matplotlib.ticker import FuncFormatter
import xml.etree.cElementTree as ET
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_color_codes()


window = 50
TRAIN_STEP = 1e5
color_cycle = sns.color_palette()
COLORS = {'ma2c': color_cycle[0], 'ia2c': color_cycle[1], 'iqll': color_cycle[2],
          'rr': color_cycle[3], 'greedy': color_cycle[4], 'ma2ceidted': color_cycle[5]}
plot_dir = 'worli/plots'


def plot_train_curve(scenario='worli'):
    cur_dir = os.path.join(scenario, 'eva_data')
    # names = ['ma2c', 'iqll']
    # labels = ['MA2C',  'IQL-LR']

    # names = ['ma2c', 'ia2c',  'iqll']
    # labels = ['MA2C', 'IA2C', 'IQL-LR']

    names = ['ma2c']
    labels = ['MA2C']

    dfs = {}
    for file in os.listdir(cur_dir):
        name = file.split('_')[0]
        print(file + ', ' + name)
        if (name in names) and (name != 'greedy'):
            df = pd.read_csv(os.path.join(cur_dir, file))
            dfs[name] = df[df.test_id == -1]

    plt.figure(figsize=(9, 6))
    ymin = []
    ymax = []

    for i, name in enumerate(names):
        if name == 'greedy':
            plt.axhline(
                y=-972.28, color=COLORS[name], linewidth=3, label=labels[i])
        else:
            df = dfs[name]
            x_mean = df.avg_reward.rolling(window).mean().values
            print(x_mean)
            x_std = df.avg_reward.rolling(window).std().values
            print(x_std)
            plt.plot(df.step.values, x_mean,
                     color=COLORS[name], linewidth=3, label=labels[i])
            ymin.append(np.nanmin(x_mean - 0.5 * x_std))
            ymax.append(np.nanmax(x_mean + 0.5 * x_std))
            plt.fill_between(df.step.values, x_mean - x_std, x_mean +
                             x_std, facecolor=COLORS[name], edgecolor='none', alpha=0.1)
    ymin = min(ymin)
    ymax = max(ymax)
    plt.xlim([0, TRAIN_STEP])
    if scenario == 'large_grid':
        plt.ylim([-1600, -400])
    else:
        plt.ylim([-1000, 0])

    def millions(x, pos):
        return '%1.1fM' % (x*1e-6)

    formatter = FuncFormatter(millions)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Training step', fontsize=18)
    plt.ylabel('Average episode reward', fontsize=18)
    plt.legend(loc='lower left', fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/%s_train.pdf' % scenario))
    plt.close()


episode_sec = 3600


def fixed_agg(xs, window, agg):
    xs = np.reshape(xs, (-1, window))
    if agg == 'sum':
        return np.sum(xs, axis=1)
    elif agg == 'mean':
        return np.mean(xs, axis=1)
    elif agg == 'median':
        return np.median(xs, axis=1)


def varied_agg(xs, ts, window, agg):
    t_bin = window
    x_bins = []
    cur_x = []
    xs = list(xs) + [0]
    ts = list(ts) + [episode_sec + 1]
    i = 0
    while i < len(xs):
        x = xs[i]
        t = ts[i]
        if t <= t_bin:
            cur_x.append(x)
            i += 1
        else:
            if not len(cur_x):
                x_bins.append(0)
            else:
                if agg == 'sum':
                    x_stat = np.sum(np.array(cur_x))
                elif agg == 'mean':
                    x_stat = np.mean(np.array(cur_x))
                elif agg == 'median':
                    x_stat = np.median(np.array(cur_x))
                x_bins.append(x_stat)
            t_bin += window
            cur_x = []
    return np.array(x_bins)


def plot_series(df, name, tab, label, color, window=None, agg='sum', reward=False):
    episodes = list(df.episode.unique())
    num_episode = len(episodes)
    num_time = episode_sec
    print(label, name)
    # always use avg over episodes
    if tab != 'trip':
        res = df.loc[df.episode == episodes[0], name].values
        for episode in episodes[1:]:
            res += df.loc[df.episode == episode, name].values
        res = res / num_episode
        print('mean: %.2f' % np.mean(res))
        print('std: %.2f' % np.std(res))
        print('min: %.2f' % np.min(res))
        print('max: %.2f' % np.max(res))
    else:
        res = []
        for episode in episodes:
            res += list(df.loc[df.episode == episode, name].values)

        print('mean: %d' % np.mean(res))
        print('max: %d' % np.max(res))

    if reward:
        num_time = 720
    if window and (agg != 'mv'):
        num_time = num_time // window
    x = np.zeros((num_episode, num_time))
    for i, episode in enumerate(episodes):
        t_col = 'arrival_sec' if tab == 'trip' else 'time_sec'
        cur_df = df[df.episode == episode].sort_values(t_col)
        if window and (agg == 'mv'):
            cur_x = cur_df[name].rolling(window, min_periods=1).mean().values
        else:
            cur_x = cur_df[name].values
        if window and (agg != 'mv'):
            if tab == 'trip':
                cur_x = varied_agg(
                    cur_x, df[df.episode == episode].arrival_sec.values, window, agg)
            else:
                cur_x = fixed_agg(cur_x, window, agg)
#         print(cur_x.shape)
        x[i] = cur_x
    if num_episode > 1:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
    else:
        x_mean = x[0]
        x_std = np.zeros(num_time)
    if (not window) or (agg == 'mv'):
        t = np.arange(1, episode_sec + 1)
        if reward:
            t = np.arange(5, episode_sec + 1, 5)
    else:
        t = np.arange(window, episode_sec + 1, window)
#     if reward:
#         print('%s: %.2f' % (label, np.mean(x_mean)))
    plt.plot(t, x_mean, color=color, linewidth=3, label=label)
    if num_episode > 1:
        x_lo = x_mean - x_std
        if not reward:
            x_lo = np.maximum(x_lo, 0)
        x_hi = x_mean + x_std
        plt.fill_between(t, x_lo, x_hi, facecolor=color,
                         edgecolor='none', alpha=0.1)
        return np.nanmin(x_mean - 0.5 * x_std), np.nanmax(x_mean + 0.5 * x_std)
    else:
        return np.nanmin(x_mean), np.nanmax(x_mean)


def plot_combined_series(dfs, agent_names, col_name, tab_name, agent_labels, y_label, fig_name, window=None, agg='sum', reward=False):
    plt.figure(figsize=(9, 6))
    ymin = np.inf
    ymax = -np.inf
    for i, aname in enumerate(agent_names):
        print(aname, tab_name)
        df = dfs[aname][tab_name]
        y0, y1 = plot_series(
            df, col_name, tab_name, agent_labels[i], COLORS[aname], window=window, agg=agg, reward=reward)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    plt.xlim([0, episode_sec])
    if (col_name == 'average_speed') and ('global' in agent_names):
        plt.ylim([0, 6])
    elif (col_name == 'wait_sec') and ('global' not in agent_names):
        plt.ylim([0, 3500])
    else:
        plt.ylim([ymin, ymax])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Simulation time (sec)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/%s.pdf' % fig_name))
    plt.close()


def sum_reward(x):
    x = [float(i) for i in x.split(',')]
    return np.sum(x)


def plot_eval_curve(scenario='worli'):
    cur_dir = os.path.join(scenario, 'eva_data')
    # names = ['ma2c', 'ia2c', 'iqll', 'greedy','fixed']
    # labels = ['MA2C', 'IA2C', 'IQL-LR', 'Greedy','Fixed-time']
    # names = ['ma2c', 'ia2c', 'iqll', 'greedy','fixed','ma2ceidted']
    # labels = ['MA2C', 'IA2C', 'IQL-LR', 'Greedy','Fixed-time','MA2C with minimum interval']
#     names = ['iqld', 'greedy']
#     labels = ['IQL-DNN','Greedy']
    # names = ['base', 'greedy', 'iqll']
    # labels = ['FixedTime', 'Greedy', 'IQLL']
    names = ['ma2c', 'rr']
    labels = ['MA2C', 'Round Robin']

    dfs = {}
    for file in os.listdir(cur_dir):
        if not file.endswith('.csv'):
            continue
        if not file.startswith(scenario):
            continue
        name = file.split('_')[1]
        measure = file.split('_')[2].split('.')[0]
        if name in names:
            print(os.path.join(cur_dir, file))
            df = pd.read_csv(os.path.join(cur_dir, file))
#             if measure == 'traffic':
#                 df['ratio_stopped_car'] = df.number_stopped_car / df.number_total_car * 100
#             if measure == 'control':
#                 df['global_reward'] = df.reward.apply(sum_reward)
            if name not in dfs:
                dfs[name] = {}
            dfs[name][measure] = df

    # plot avg queue
    plot_combined_series(dfs, names, 'avg_queue', 'traffic', labels,
                         'Average queue length (veh)', scenario + '_queue', window=60, agg='mv')
    # plot avg speed
    plot_combined_series(dfs, names, 'avg_speed_mps', 'traffic', labels,
                         'Average car speed (m/s)', scenario + '_speed', window=60, agg='mv')
    # plot avg waiting time
    plot_combined_series(dfs, names, 'avg_wait_sec', 'traffic', labels,
                         'Average intersection delay (s/veh)', scenario + '_wait', window=60, agg='mv')
    # plot trip completion
    plot_combined_series(dfs, names, 'number_arrived_car', 'traffic', labels,
                         'Trip completion rate (veh/5min)', scenario + '_tripcomp', window=300, agg='sum')
    # plot trip time
#     plot_combined_series(dfs, names, 'duration_sec', 'trip', labels,
#                          'Avg trip time (sec)', scenario + '_triptime', window=60, agg='mean')
#     plot trip waiting time
    plot_combined_series(dfs, names, 'wait_sec', 'trip', labels,
                         'Avg trip delay (s)', scenario + '_tripwait', window=60, agg='mean')

    plot_combined_series(dfs, names, 'reward', 'control', labels,
                         'Step reward', scenario + '_reward', reward=True, window=6, agg='mv')


if __name__ == '__main__':
    plot_eval_curve()
