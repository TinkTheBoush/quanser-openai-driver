# Quanser OpenAI Driver
Has an OpenAI Gym wrapper for the Quanser Qube Servo 2 and Quanser Aero

- [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Recompiling Cython](#recompiling-cython-code)
- [Basic Usage](#usage)
- [Warning](#warning)
    - [Quick fix](#the-fastest-solution-no-restart-required-is-to-remove-the-semaphore-of-the-board)


# Setup
We have tested on Ubuntu 16.04 LTS and Ubuntu 18.04 LTS using Python 2.7 and Python 3.6.5<br>


### Prerequisites
Install the HIL SDK from Quanser.<br>
A mirror is available at https://github.com/BlueRiverTech/hil_sdk_linux_x86_64.<br>

You can install the driver by:
```bash
    git clone https://github.com/BlueRiverTech/hil_sdk_linux_x86_64.git
    sudo ./hil_sdk_linux_x86_64/setup_hil_sdk
```

You also must have pip installed:
```bash
    sudo apt-get install python3-pip
```

**Note:** this requires a version of the HIL SDK that supports buffer overwrite on overflow (circular buffers).<br>
(The mirror posted above supports buffer overflow.)<br>


### Installation
We recommend that you use a virtual environment such as [virtualenv](https://virtualenv.pypa.io/en/stable/), [conda](https://conda.io/docs/user-guide/getting-started.html), or [Pipenv](https://pipenv.readthedocs.io/en/latest/)

You can install the driver by cloning and pip-installing:
```bash
    git clone https://github.com/BlueRiverTech/quanser-openai-driver.git
    cd quanser-openai-driver
    pip3 install -e .
```

Once you have that setup: Run the classical control baseline (ensure the Qube is connected to your computer)<br>
```bash
python tests/test.py --env QubeFlipUpEnv --control flip-up
```


# Usage
Usage is very similar to most OpenAI gym environments but **requires** that you close the environment when finished.
Without safely closing the Env, bad things may happen. Usually you will not be able to reopen the board.

This can be done with a context manager using a `with` statement
```python
import gym
from gym_brt import QubeInvertedPendulumEnv

num_episodes = 10
num_steps = 250

with QubeInvertedPendulumEnv() as env:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
```

Or can be closed manually by using `env.close()`. You can see an [example here](docs/alternatives.md#usage).


# Environments
Information about various environments can be found in [docs/envs](docs/envs.md).


# Control
Information about baselines can be found in [docs/control](docs/control.md).


# Hardware Wrapper
Information about the Python wrapper for Quanser hardware and Qube Servo 2 simulator can be found in [docs/quanser](docs/quanser.md).


# Warning
Forgetting to close the environment or incorrectly closing the env leads to several possible issues. The worst including segfaults.

The most common case when the env was not properly closed: you can not reopen the env and you get:
```
Error 1073: QERR_BOARD_ALREADY_OPEN: The HIL board is already opened by another process. The board does not support access from more than one process at the same time.
```

#### The fastest solution (no restart required) is to remove the semaphore of the board:
1. Find the location of the semaphore by `ls /dev/shm`.
    - It will start with `sem.qube_servo2_usb$xxxxxxxxxxxxx` (with the 'x's being some alphanumeric sequence).
1. Use the `rm` command to remove it.
    - Ex: `rm /dev/shm/sem.qube_servo2_usb$xxxxxxxxxxxxx`

