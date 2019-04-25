import numpy as np
import gym, glob, tictactoe_env

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def create_mpl(input_shape, output_neurons):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Dense(output_neurons))
    model.add(Activation('linear'))
    return model

def get_weights_path(weights_name):
    path = None
    for f in glob.glob(weights_name):
        if(f != ""):
            path = f
            break
    return path

def main():
    dqn_weights_name = "tic_tac_toe_dqn_agent.h5f"
    #dqn_2_weights_name = "tic_tac_toe_dqn2_agent.h5f"
    ENV_NAME = 'tictactoe-v0'
    env = gym.make(ENV_NAME)
    mlp = create_mpl(env.get_obs_space().shape, env.get_action_space().n)
    memory = SequentialMemory(limit = 5000000, window_length = 1)
    policy = BoltzmannQPolicy()

    # First Agent
    dqn_1 = DQNAgent(model = mlp, memory = memory, policy = policy, nb_actions = env.get_action_space().n, nb_steps_warmup = 32, gamma = 0.3)
    dqn_1.compile(Adam(lr=1e-3), metrics=['accuracy', 'mae'])

    # First Agent Training
    for i in range (0, 10):
        env.reset_all()
        w_path = get_weights_path(dqn_weights_name)
        if (w_path != None):
            dqn_1.load_weights(w_path)
        dqn_1.fit(env, nb_steps = 3000, visualize=True, verbose=2)
        dqn_1.save_weights(dqn_weights_name, overwrite=True)

if __name__ == '__main__':
    main()