This is an algorithm that uses deep RL techniques to solve mazes with variable difficulty.

#### random environment.py
It creates a random environment every time the Environment class is instantiated.

#### agent.py
There are currently five functions, which are used by train and test.py to allow the agent to interact with the environment. 
This is the script which will be used to train, and then test, the agent. It will create a new random environment, train the agent for 10 minutes in this environment, and then test the agent’s greedy policy in this environment with an episode of 100 steps.

If you run python3 train and test.py from the command line, you should see the environment being displayed. The red circle is the agent’s current state, and the blue circle is the goal. The light region is free space, which the agent can move through, and the dark region is an obstacle, which the agent cannot move through. If the agent tries to move into the obstacle, the agent will remain in its current state. The agent must navigate the “maze” and reach the goal as quickly as possible.


