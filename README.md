This repo is made by forking reference repository [cts198859/deeprl_signal_control](https://github.com/cts198859/deeprl_signal_control). Based on reference repository, we built a simulation environment of Worli.

## Requirements

* Python3==3.5
* [Tensorflow](http://www.tensorflow.org/install)==1.12.0
* [SUMO](http://sumo.dlr.de/wiki/Installing)>=1.1.0

## Usages

First define all hyperparameters in a config file under `[config_dir]`, and create the base directory of experiements `[base_dir]`. Before training, please call `build_file.py` under `[environment_dir]/data/` to generate SUMO network files for `small_grid` and `large_grid` environments.

1. To train a new agent, run

```
python3 main.py --base-dir [base_dir]/[agent] train --config-dir [config_dir] --test-mode no_test
```

`[agent]` is from `{ia2c, ma2c, iqll, iqld}`. `no_test` is suggested, since tests will significantly slow down the training speed.

2. To access tensorboard during training, run

```
tensorboard --logdir=[base_dir]/log
```

3. To evaluate and compare trained agents, run

```
python3 main.py --base-dir [base_dir] evaluate --agents [agents] --evaluation-seeds [seeds]
```

Evaluation data will be output to `[base_dir]/eva_data`, and make sure evaluation seeds are different from those used in training. Under default evaluation setting, the inference policy of A2C is stochastic whereas that of Q-learning is greedy (deterministic). To explicitly specifiy the inference policy type, pass argument `--evaluation-policy-type [default/stochastic/deterministic]`. Please note running a determinisitc inference policy for A2C may cause the performance loss, due to the violation of "on-policy" learning.

4. To visualize the agent behavior, run

```
python3 main.py --base-dir [base_dir] evaluate --agents [agent] --evaluation-seeds [seed] --demo
```
