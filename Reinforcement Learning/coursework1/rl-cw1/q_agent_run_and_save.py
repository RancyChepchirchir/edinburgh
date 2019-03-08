if __name__ == "__main__":

    from q_agent import *

    # 4, 7 so far are best

    # Failed combos:
    # > 6, 7
    # > 6, 6
    # > 6, 4
    # > 4, 9 is horrible
    # > 4, 6 is horrible

    # Okay combos:
    # > 6, 5 does almost as good as 4, 7

    forward_visibility = 4  # max 10
    side_visibility = 7  # max 9

    learn = True
    use_learned = True
    learning_path = os.path.join('run', run_subdir(forward_visibility, side_visibility), 'state_action_reward_dict.p')
    if use_learned and os.path.exists(learning_path):
        with open(learning_path, 'r') as input_file:
            state_action_reward_dict = pickle.load(input_file)
    else:
        state_action_reward_dict = None
    a = QAgent(state_action_reward_dict, forward_visibility, side_visibility, False)
    a.run(learn, episodes=1001, draw=True)
    print 'Total reward: ' + str(a.total_reward)
