import numpy as np
# Q
q = np.matrix(np.zeros([6, 6]))
# Reward
r = np.matrix([[-1, -1, -1, -1,  0,  -1],
[-1, -1, -1,  0, -1, 100],
[-1, -1, -1,  0, -1,  -1],
[-1,  0,  0, -1,  0,  -1],
[ 0, -1, -1,  0, -1, 100],
[-1,  0, -1, -1,  0, 100]])
gamma = 0.8
epsilon = 0.4
# the main training loop
for episode in range(101):
  # random initial state
  state = np.random.randint(0, 6)
  # if not final state
  while (state != 5):
    # choose a possible action
    # Even in random case, we cannot choose actions whose r[state, action] = -1.
    possible_actions = []
    possible_q = []
    for action in range(6):
      if r[state, action] >= 0:
        possible_actions.append(action)
        possible_q.append(q[state, action])
        # Step next state, here we use epsilon-greedy algorithm.
        action = -1
        if np.random.random() < epsilon:
          # choose random action
          action = possible_actions[np.random.randint(0, len(possible_actions))]
      else:
        # greedy
        action = possible_actions[np.argmax(possible_q)]
        # Update Q value
        q[state, action] = r[state, action] + gamma * q[action].max()
        # Go to the next state
        state = action
        # Display training progress
if episode % 10 == 0:
  print("------------------------------------------------")
  print("Training episode: %d" % episode)
print(q)
