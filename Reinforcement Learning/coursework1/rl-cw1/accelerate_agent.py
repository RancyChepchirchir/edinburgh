import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState

import os
import csv
import numpy as np
import pickle

class AccelerateAgent(Agent):
    def __init__(self):
        super(AccelerateAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0

        self.this_episode = -1
        self.run_dir = os.path.join('run', 'accelerate_agent')
        self.reward_path = os.path.join(self.run_dir, 'total_reward.txt')

        # Remove rewards file. Can't do this in initialisation, which runs every episode.
        if os.path.exists(self.reward_path):
            os.remove(self.reward_path)

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward_last_episode = self.total_reward  # for logging; otherwise lose this next episode
        self.total_reward = 0

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        self.total_reward += self.move(Action.ACCELERATE)

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        def save_data(e):
            """ Write learning data (dict to pickle object) and episode final reward (to CSV)
            """
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

if __name__ == "__main__":
    a = AccelerateAgent()
    a.run(True, episodes=101, draw=True)
    print 'Total reward: ' + str(a.total_reward)
