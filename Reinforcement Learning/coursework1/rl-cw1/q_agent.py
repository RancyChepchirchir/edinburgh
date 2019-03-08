import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState

import os
import csv
import numpy as np
import pickle

def run_subdir(forward_visibility, side_visibility):
    return 'f{}_s{}'.format(forward_visibility, side_visibility)

class QAgent(Agent):
    def __init__(self,
                 state_action_reward_dict=None,
                 forward_visibility=4,
                 side_visibility=7,
                 fast_beginning_learning=True,
                 use_symmetry=True):

        super(QAgent, self).__init__()

        self.total_reward = 0

        # Define Q-learning parameters.
        self.epsilon = 0.1  # choose non-optimal (i.e., explore) this proportion of the time
        self.alpha = 0.01  # learning rate; set to one to forget everything before last iteration
        self.gamma = 0.9

        # Set arguments and related variables.
        self.initialise_forward_motion = state_action_reward_dict is None
        if state_action_reward_dict is None:
            self.state_action_reward_dict = {}  # key: (state, action); value: (reward, num times Q(s,a) has been seen)
        else:
            self.state_action_reward_dict = state_action_reward_dict
        self.forward_visibility = forward_visibility
        self.side_visibility = side_visibility
        self.fast_beginning_learning = fast_beginning_learning
        self.use_symmetry = use_symmetry

        # Save temporary state and reward data.
        self.state = None
        self.new_state = None
        self.reward = None

        # Set variables only used in callback.
        self.max_iteration = None
        self.total_reward_last_episode = None
        self.this_episode = -1

        # Define paths.
        self.run_dir = os.path.join('run', run_subdir(self.forward_visibility, self.side_visibility))
        self.learning_path = os.path.join(self.run_dir, 'state_action_reward_dict.p')
        self.reward_path = os.path.join(self.run_dir, 'total_reward.txt')

        # Remove rewards file. Can't do this in initialisation, which runs every episode.
        if os.path.exists(self.reward_path):
            os.remove(self.reward_path)


    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode.
        self.total_reward_last_episode = self.total_reward  # for logging; otherwise lose this next episode
        self.total_reward = 0

        self.state = self.define_state(grid)

        # Initialize with small reward for acceleration.
        if self.initialise_forward_motion and self.new_state is None:  # new_state is None in first episode
            self.state_action_reward_dict[(self.state, Action.ACCELERATE)] = (1, 1)

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        def choose_action(state):
            """ Choose action using epsilon-greedy approach. Cannot be None.
            """
            best_Q = None
            best_action = None
            shuffled_actions = self.getActionsSet()  # shuffle in next line...
            np.random.shuffle(shuffled_actions)
            for action in shuffled_actions:
                try:
                    Q, _ = self.state_action_reward_dict[(state, action)]
                except KeyError:
                    # If haven't tried action in this state, then try it.
                    return action
                if Q > best_Q:
                    best_Q = Q
                    best_action = action
            if np.random.uniform() < self.epsilon:
                exploration_actions = [a for a in self.getActionsSet() if a != best_action]
                best_action = np.random.choice(exploration_actions)
            return best_action

        # Choose action given state.
        self.action = choose_action(self.state)

        # Take action, observe new state.
        self.reward = self.move(self.action)
        self.total_reward += self.reward


    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        self.new_state = self.define_state(grid)


    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """

        def get_max_Q(state):
            """ Get max Q over all actions for a state. Can be None.
            """
            best_Q = None
            for action in self.getActionsSet():
                try:
                    Q, _ = self.state_action_reward_dict[(state, action)]
                except KeyError:
                    continue
                if Q > best_Q:
                    best_Q = Q
            return best_Q

        # Get max_a'{Q(s',a')}.
        new_state_max_Q = get_max_Q(self.new_state)
        if new_state_max_Q is None:
            new_state_max_Q = 0.
        try:
            Q, iter_num = self.state_action_reward_dict[(self.state, self.action)]
        except KeyError:
            Q, iter_num = 0., 0
        
        iter_num += 1  # only useful for changing alpha (self.fast_beginning_learning = True)

        if self.fast_beginning_learning:
            # Big alpha values in beginning.
            alpha_iter = max(1/float(iter_num), self.alpha)
        else:
            alpha_iter = self.alpha

        # Update Q.
        new_Q = Q + alpha_iter*(self.reward + self.gamma*new_state_max_Q - Q)
        self.state_action_reward_dict[(self.state, self.action)] = (new_Q, iter_num)
        if self.use_symmetry:
            # Add Q data for state inverted over the y-axis (left to right).
            # Make sure to invert action as well.
            # If flipped state is the same, this just repeats the non-flipped Q update.
            flipped_state = self.flip_state(self.state)
            flipped_action = self.flip_action(self.action)
            self.state_action_reward_dict[(flipped_state, flipped_action)] = (new_Q, iter_num)

        self.state = self.new_state


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        def save_data(e):
            """ Write learning data (dict to pickle object) and episode final reward (to CSV)
            """
            with open(self.learning_path, 'wb') as handle:
                pickle.dump(self.state_action_reward_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            write_mode = 'w' if not os.path.exists(self.reward_path) else 'a'
            with open(self.reward_path, write_mode) as f:
                f.write('{},{}\n'.format(e, self.total_reward_last_episode))
            return True

        if episode != self.this_episode and episode > 1:
            save_data(episode-1)
            self.this_episode = episode

        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

        # self.print_learning_data()  # only run this if you restrict the number of frames to be just a handful!


    # #####
    # Helper functions
    # #####
    def define_state(self, grid):
        """ Subset the grid array for cells around the player's car and represent as string.
        """
        player_number = 2  # number representing player's car in grid
        player_index = list(grid[0,:]).index(player_number)
        x_min = max(player_index - self.side_visibility, 0)
        x_max = min(player_index + self.side_visibility, grid.shape[1]-1)
        y_max = self.forward_visibility
        return self.array2str(grid[:(y_max+1), x_min:(x_max+1)])


    def array2str(self, array):
        """ Get hashable state representation.
        """
        return str(array.shape) + ''.join(str(x) for x in array.ravel())


    def str2array(self, string):
        """ Recover array representation.
        """
        shape, array = [s.split('(')[-1] for s in string.split(')')]
        array = [int(s) for s in array]
        shape = [int(s) for s in shape.split(', ')]
        return np.array(array).reshape(shape)


    def flip_action(self, action):
        """ Flip LEFT (12) and RIGHT (11) actions. Don't change ACCELERATE and BREAK.
        """
        return {11: 12, 12: 11}.get(action, action)


    def flip_state(self, state):
        """ Return state reflected over y-axis.
        """
        return self.array2str(np.fliplr(self.str2array(state)))


    def print_learning_data(self):
        """ Print state_action_reward_dict in nice format.
        """
        for key, value in self.state_action_reward_dict.iteritems():
            print 'state:\n{}\naction: {}\nQ: {}\nn: {}'.format(\
                self.str2array(key[0]), key[1], value[0], value[1])

if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=2, draw=True)
    print 'Total reward: ' + str(a.total_reward)
