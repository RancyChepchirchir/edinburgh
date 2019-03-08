import cv2
import numpy as np
import os
import csv
import pickle
from collections import OrderedDict

from enduro.agent import Agent
from enduro.action import Action


def run_subdir(string):
    return '{}'.format(string)

class FunctionApproximationAgent(Agent):
    def __init__(self,
                 weights=None,
                 watch_game=False):
        super(FunctionApproximationAgent, self).__init__()

        self.watch_game = watch_game
        self.weights = weights  # initialize in self.act since you need to know length of features vector
        self.weights_all = []

        self.total_reward = 0

        # Define Q-learning parameters.
        self.epsilon = 0.1  # choose non-optimal (i.e., explore) this proportion of the time
        self.alpha = 0.00025  # learning rate; set to one to forget everything before last iteration
        self.gamma = 0.9

        # Save temporary state and reward data.
        # self.state_action = None
        # self.new_state_action = None
        self.reward = None

        self.run_dir = os.path.join('run', 'initial')
        self.weights_path = os.path.join(self.run_dir, 'weights.p')
        self.weights_all_path = os.path.join(self.run_dir, 'weights_all.p')
        self.reward_path = os.path.join(self.run_dir, 'total_reward.txt')

        # Set variables only used in callback.
        self.max_iteration = None
        self.total_reward_last_episode = None
        self.this_episode = -1

        # Remove rewards file. Can't do this in initialisation, which runs every episode.
        if os.path.exists(self.reward_path):
            os.remove(self.reward_path)


    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        # Reset the total reward for the episode.
        self.total_reward_last_episode = self.total_reward  # for logging; otherwise lose this next episode
        self.total_reward = 0

        self.road = road
        self.cars = cars
        self.speed = speed
        self.grid = grid

        self.build_state_action(road, cars, speed, grid, Action.NOOP)

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)


    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        best_Q = None
        self.action = None  # best action
        shuffled_actions = self.getActionsSet()  # shuffle in next line...
        np.random.shuffle(shuffled_actions)
        for action in shuffled_actions:
            state_action = self.build_state_action(self.road, self.cars, self.speed, self.grid, action)
            if self.weights is None:
                # self.weights = np.random.normal(0, 0.0001, len(state_action)) # {action: list of weights}
                self.weights = np.zeros(len(state_action)) # {action: list of weights}
                for key in [
                    'action_accel'
                    # 'toward_center',
                    # 'close_to_bump_left_or_right'
                    ]:
                    try:
                        idx = state_action.keys().index(key)
                        self.weights[idx] = 0.1
                    except ValueError:
                        continue
                for key in [
                    # 'close_to_bump_accel',
                    # 'close_to_bump'
                    ]:
                    try:
                        idx = state_action.keys().index(key)
                        self.weights[idx] = -0.05
                    except ValueError:
                        continue
                # self.weights[[6, 9]] = 0.05
                # self.weights[[5, 8]] = -0.05
            Q = self.calc_Q(state_action, self.weights)
            if Q > best_Q:
                best_Q = Q
                self.action = action
        if np.random.uniform() < self.epsilon:
            exploration_actions = [a for a in self.getActionsSet() if a != self.action]
            self.action = np.random.choice(exploration_actions)
        self.reward = self.move(self.action)
        self.total_reward += self.reward

        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work


    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        self.road2 = road
        self.cars2 = cars
        self.speed2 = speed
        self.grid2 = grid

        # print 'self.road:\n{}\n'.format(self.road)
        # print 'self.cars:\n{}\n'.format(self.cars)
        # print 'self.speed:\n{}\n'.format(self.speed)
        # print 'self.grid:\n{}\n'.format(self.grid)
        # print 'self.state_action:\n{}\n'.format(self.state_action)
        # print 'weights:\n{}\n'.format(self.weights)


    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        # Calculate max Q_{t}(s_{t+1}, a_{t+1})
        max_Q = None
        for action in self.getActionsSet():
            state_action = self.build_state_action(self.road2, self.cars2, self.speed2, self.grid2, action)
            Q_try = self.calc_Q(state_action, self.weights)
            if Q_try > max_Q:
                max_Q = Q_try

        # Calculate Q_t
        self.state_action = self.build_state_action(self.road, self.cars, self.speed, self.grid, self.action)
        # print self.state_action
        Q = self.calc_Q(self.state_action, self.weights)

        # Update weights
        self.weights += self.alpha*(self.reward + self.gamma*max_Q - Q) * np.array(self.state_action.values())

        self.road = self.road2
        self.cars = self.cars2
        self.speed = self.speed2
        self.grid = self.grid2


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        def save_data(e):
            """ Write learning data (dict to pickle object) and episode final reward (to CSV)
            """
            with open(self.weights_path, 'wb') as handle:
                pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.weights_all_path, 'wb') as handle:
                pickle.dump(self.weights_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
            write_mode = 'w' if not os.path.exists(self.reward_path) else 'a'
            with open(self.reward_path, write_mode) as f:
                f.write('{},{}\n'.format(e, self.total_reward_last_episode))
            return True

        if episode != self.this_episode and episode > 1:
            self.weights_all.append(self.weights.copy())
            save_data(episode-1)
            self.this_episode = episode
            print "\n=====\nepisode {}: {}\n=====\n".format(episode, self.total_reward_last_episode)
            for i, key in enumerate(self.state_action.keys()):
                print '{}: {:0.4g}'.format(key, self.weights[i])

        # if iteration % 1000 == 0:
        #     print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
            # print 'self.road:\n{}\n'.format(self.road)
            # print 'self.cars:\n{}\n'.format(self.cars)
            # print 'self.speed:\n{}\n'.format(self.speed)
            # print 'self.grid:\n{}\n'.format(self.grid)
            # print 'self.state_action:\n{}\n'.format(self.state_action)
            # print 'weights:\n{}\n'.format(self.weights)


        # You could comment this out in order to speed up iterations
        if self.watch_game:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)


    def build_state_action(self, road, cars, speed, grid, action):

        # Calculate curve.
        top_mean = np.mean(road[0][5][0])
        bottom_mean = np.mean(road[11][5][0])
        curve_val = top_mean - bottom_mean  # > 0 is curve right, < 0 is curve left
        toward_curve = int((action == 11 and curve_val > 20.) or (action == 12 and curve_val < 20.))
        away_from_curve = int((action == 11 and curve_val < 20.) or (action == 12 and curve_val > 20.))

        # Get player location. Note that 11 is RIGHT, 12 is LEFT.
        midpoint = 4.5
        player_location = np.argwhere(grid[0, :] == 2)[0][0] - midpoint  # negative is left of center, positive is right
        dist_from_center = abs(player_location)
        toward_center = int((action == 11 and player_location < 0.) or (action == 12 and player_location > 0.))
        away_from_center = int((action == 11 and player_location > 0.) or (action == 12 and player_location < 0.))

        # Calcuate features regarding other car locations.
        def get_other_car_features(other, action, max_ydist=6, max_xdist=3):
            if other is not None:
                car_ydist, car_xdist = other
                toward_car = int((action == 11 and car_xdist > 0) or (action == 12 and car_xdist < 0))
                away_from_car = int((action == 11 and car_xdist <= 0) or (action == 12 and car_xdist >= 0))
                # print 'toward_car: {}'.format(toward_car)
                # print 'away_from_car: {}'.format(away_from_car)
                # Cap distance values.
                car_ydist_clean = min(max_ydist, car_ydist) if abs(car_xdist) <= max_xdist else max_ydist
                car_xdist_clean = min(max_xdist, abs(car_xdist)) if car_xdist <= max_ydist else max_xdist
                car_xdist, car_ydist = car_xdist_clean, car_ydist_clean
            else:
                car_xdist = max_xdist
                car_ydist = max_ydist
                toward_car = 0
                away_from_car = 0
            return car_ydist, car_xdist, toward_car, away_from_car
        others = self.get_relative_car_coords(grid)
        car1_ydist, car1_xdist, toward_car1, away_from_car1 = get_other_car_features(others[0], action)
        car2_ydist, car2_xdist, toward_car2, away_from_car2 = get_other_car_features(others[1], action)

        # Penalize/reward actions directly.
        action_accel = int(action == 1)
        action_break = int(action == 5)
        action_left_or_right = int(action == 11 or action == 12)
        # close_to_bump = int(((car1_ydist <= 3 and car1_xdist <= 2) or (car2_ydist <= 3 and car2_xdist <= 2)) and speed > 0.)
        close_to_bump = int(car1_ydist <= 2 and car1_xdist <= 1 and speed > 0.)

        state_action = OrderedDict()
        # state_action['toward_curve'] = toward_curve
        # state_action['away_from_curve'] = away_from_curve
        state_action['dist_from_center'] = dist_from_center
        state_action['toward_center'] = toward_center
        state_action['away_from_center'] = away_from_center
        # state_action['speed'] = speed
        state_action['action_accel'] = int((not close_to_bump) and action_accel)
        # state_action['action_break'] = action_break
        state_action['car1_ydist'] = car1_ydist
        # state_action['car1_ydist_sq'] = car1_ydist**2
        state_action['car1_xdist'] = car1_xdist
        state_action['toward_car1'] = toward_car1
        state_action['away_from_car1'] = away_from_car1
        # state_action['close_to_bump'] = close_to_bump
        state_action['close_to_bump_accel'] = int(close_to_bump and action_accel)
        state_action['close_to_bump_left_or_right'] = int(close_to_bump and action_left_or_right)
        # # state_action['toward_center_and_away_from_car1'] = int(state_action['toward_center'] and state_action['away_from_car1'])
        # state_action['car2_ydist'] = car2_ydist
        # # # state_action['car2_ydist_sq'] = car2_ydist**2
        # state_action['car2_xdist'] = car2_xdist
        state_action['toward_car2'] = toward_car2
        state_action['away_from_car2'] = away_from_car2

        return state_action


    def get_relative_car_coords(self, grid, cars_to_keep=2):

        # Find car coordinates.
        self_ = np.argwhere(grid == 2)[0]
        others_all = np.argwhere(grid == 1)

        # Only keep closest cars.
        dists = [(other, self.distance(self_, other)) for other in others_all]
        others = [tup[0] for tup in sorted(dists, key=lambda tup: tup[1])[:cars_to_keep]]

        # Add None values if not enough cars.
        others_relative = [other - self_ for other in others]
        cars_to_add = cars_to_keep - len(others_relative)
        for _ in range(cars_to_add):
            others_relative.append(None)

        return others_relative


    def distance(self, p0, p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


    def calc_Q(self, state_action, weights):
        return np.dot(np.array(state_action.values()), weights)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
