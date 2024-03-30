############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
from collections import deque
import random
from matplotlib import pyplot as plt ### should remove it bc it's not standard library

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 900
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        #Initialise dqn
        self.dqn = DQN()
        #Initialise replay_buffer
        self.replay_buffer = ReplayBuffer()
        #Initialise minibatch size
        self.minibatch_size = 500
        #Counter to update update_q_target_network
        self.K = 0
        #Set the exploration parameters
        self.epsilon = 0.9
        self.epsilon_reset = 0.9
        #Initialise counter for number of Episodes
        self.number_of_episode = 0
        #Initialise losses for Plots  ---- to be removed
        self.losses = []
        #Initialise loss plot --------- to be removed
        self.loss_plot = Loss_plots()
        #self.evaluate_greedy_policy = False
        self.reached = True
        self.wall_after_goal = False
        self.use_epsilon_greedy = False
        #self.epsilon_greedy = 0.9
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.loss_plot.add_loss((sum(self.losses)/self.episode_length), self.number_of_episode)
            print('Iteration ' + str(self.number_of_episode) + ', Loss = ' + str((sum(self.losses)/self.episode_length)))
            self.losses = []    #to be removed befoe submission
            self.number_of_episode += 1
            self.K += 1
            self.reached = True
            self.wall_after_goal = False
            if self.K % 5 == 0:
                # if self.epsilon > 0.5:
                self.dqn.update_q_target_network()
            # if self.K % 15 == 0:
            #     if self.epsilon < 0.5:
            #         self.dqn.update_q_target_network()
            if (self.K+8) % 5  == 0:
                if self.episode_length > 200:
                    self.episode_length = self.episode_length - 100
            self.use_epsilon_greedy = False
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        if self.num_steps_taken % self.episode_length == 0 and (self.K+6) % 40 == 0:
            if self.epsilon > 0.31:
                self.epsilon = self.epsilon - 0.075

            print("Epsilon is: ", self.epsilon)
            # if self.episode_length > 101:
            #     self.episode_length = self.episode_length - 100
            print("The episode_length is:", self.episode_length)
            # if  self.minibatch_size > 800:
            #     self.minibatch_size = self.minibatch_size -200
            print("The minibatch size is:", self.minibatch_size)
            #self.evaluate_greedy_policy = False
            #self.evaluate_greedy_policy = True
            #self.epsilon = 0
        if self.K > 350 and self.epsilon > 0.2:
            self.epsilon = self.epsilon - 0.01

        if self.use_epsilon_greedy == True:
            self.epsilon = 0.9
            if self.K > 30 and self.K % 5 == 0:
                self.epsilon = self.epsilon - 0.2

        state_tensor = torch.tensor(state)
        predicted_q_values = self.dqn.q_network(state_tensor)
        predicted_q_values = predicted_q_values.detach().numpy()
        action = np.argmax(predicted_q_values)


        action_probability = np.ones(3) * (self.epsilon / 3)
        action_probability[action] += (1 - self.epsilon)
        #action_probability[2] = 0

        action = random.choices([0,1,2], action_probability)
        action = action[0]
        continuous_action = self.discrete_to_continuous(action)

        self.state = state
        self.num_steps_taken += 1
        self.action = action
        return continuous_action
        # # Here, the action is random, but you can change this
        # action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # # Update the number of steps which the agent has taken
        # self.num_steps_taken += 1
        # # Store the state; this will be used later, when storing the transition
        # self.state = state
        # # Store the action; this will be used later, when storing the transition
        # self.action = action
        # return action

    def discrete_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1: #Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        # if discrete_action == 2: #Move left
        #     continuous_action = np.array([-0.02, 0], dtype=np.float32)
        if discrete_action == 2: #Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        if self.state[0] == next_state[0] and self.state[1] == next_state[1] and self.wall_after_goal == False:
            hit_wall = 0
        else: hit_wall = 1


        norm_steps_taken = ((self.num_steps_taken % self.episode_length)-100)/40000


        reward = (1 - distance_to_goal - norm_steps_taken) * hit_wall

        # Create a transition
        transition = (self.state, self.action, reward, next_state)


        # Now you can do something with this transition ...
        self.train_agent(transition)

        if distance_to_goal < 0.2:
            self.use_epsilon_greedy = True

        if distance_to_goal < 0.03 :
            if self.reached == True:
                print("The number of steps taken are:", (self.num_steps_taken-1)%self.episode_length)
                self.reached = False
                self.wall_after_goal = True
                # if self.K > 8:
                #     self.episode_length = ((self.num_steps_taken-1)%self.episode_length) + 75

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predicted_q_values = self.dqn.q_network(input)
        predicted_q_values = predicted_q_values.detach().numpy()
        action = np.argmax(predicted_q_values)
        action = self.discrete_to_continuous(action)
        return action

    def train_agent(self, transition):
        self.replay_buffer.add_transition(transition)
        if len(self.replay_buffer.buffer) > self.minibatch_size and self.K > 6:
            minibatch, indexes_from_minibatch = self.replay_buffer.sample_minibatch(self.minibatch_size)
            loss = self.dqn.train_q_network(minibatch)
            self.replay_buffer.update_minibatch_weights(indexes_from_minibatch, loss)
            self.losses.append(loss)

    #Function to be removed before submission because it modifies the train_and_test file
    def plot_loss(self):
        return self.loss_plot.plot()

class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        #layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_2_output)
        return output

class ReplayBuffer:
    buffer = deque(maxlen = 5000)
    weights = deque(maxlen=5000)
    probabilities = deque(maxlen=5000)
    epsilon = 0.001
    alpha = 0.6

    def add_transition(self, transition):
        # if self.check_transition(transition)==False:
        #     return
        # else:
        self.buffer.append(transition)
        self.assign_weight(transition)

        #print(self.buffer)

    def check_transition(self, transition):
        for i in range(len(self.buffer)):
            if transition[0].all()==self.buffer[i][0].all() and transition[1]==self.buffer[i][1] and transition[2]==self.buffer[i][2] and transition[3].all()==self.buffer[i][3].all():
                return False
        else:
            return True

    def sample_minibatch(self, minibatch_size):
        probability = (np.power(self.weights, self.alpha))/ (np.sum(np.power(self.weights,self.alpha)))

        indexes = list(range(0, len(self.buffer)))
        indexes_from_minibatch = random.choices(indexes, probability, k=minibatch_size)
        # print(len(indexes))
        # print(len(probability))
        minibatch = []
        for i in range(len(indexes_from_minibatch)):
            minibatch.append(self.buffer[indexes_from_minibatch[i]])
        return minibatch, indexes_from_minibatch

    def assign_weight(self, transition):
        if not self.weights:
            weight = self.epsilon
            self.weights.append(weight)
        else:
            weight = np.max(self.weights)
            self.weights.append(weight)

    def update_minibatch_weights(self, indexes_from_minibatch, loss):
        for i in range(len(indexes_from_minibatch)):
            self.weights[indexes_from_minibatch[i]] = self.epsilon + abs(loss)

class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        self.q_target = self.q_network
        #weigths = self.q_network.torch.nn.Module.state_dict()
        self.q_target.load_state_dict(self.q_network.state_dict())
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch):
        minibatch_state_inputs = [i[0] for i in minibatch]
        rewards = self.calculate_Bellman_reward(minibatch)
        #minibatch_reward_inputs = [i[2] for i in minibatch]
        minibatch_actions_inputs = [i[1] for i in minibatch]

        minibatch_state_tensor = torch.tensor(minibatch_state_inputs, dtype=torch.float32)
        #minibatch_reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        minibatch_actions_tensor = torch.tensor(minibatch_actions_inputs)
        predicted_q_value_tensor = self.q_network.forward(minibatch_state_tensor).gather(dim=1, index=minibatch_actions_tensor.unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, rewards)
        return loss

    def get_Q_values(self):
        q_values = np.zeros([10, 10, 4])
        for col in range(10):
            for row in range(10):
                x = (col / 10.0) + 0.05
                y =  (row / 10.0) + 0.05
                input = torch.tensor((x,y), dtype=torch.float32).unsqueeze(0)
                prediction_q_values = self.q_network.forward(input)

                for action in range(4):
                    q_values[col, row, action] = prediction_q_values[0][action]
        return q_values

    def get_max_Q_value(self, state):
        input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predicted_q_values = self.q_network.forward(input)
        predicted_q_values = predicted_q_values.detach().numpy()
        max_Q_value = np.argmax(predicted_q_values)
        return max_Q_value

    def calculate_Bellman_reward(self, minibatch):
        r = torch.tensor([i[2] for i in minibatch], dtype=torch.float32)
        # for i in minibatch:
        #     if i[0][0] == i[3][0] and i[0][1] == i[3][1]:
        #         r = torch.tensor(i[2] - 2, dtype=torch.float32)
        #     else:
        #         r = torch.tensor(i[2], dtype=torch.float32)

        state_tensor = torch.tensor([i[3] for i in minibatch], dtype=torch.float32)
        gamma = torch.ones(len(minibatch)) * 0.9

        with torch.no_grad():
            q_target = torch.argmax(self.q_target(state_tensor), dim=1)
            double_q_net = self.q_network(state_tensor).gather(dim=1, index=q_target.unsqueeze(-1)).squeeze(-1)
            rewards = r + gamma*double_q_net
        # gamma = torch.ones(100) * 0.9
        # rewards = []
        #
        # r = torch.tensor([i[2] for i in minibatch], dtype=torch.float32)
        # state_tensor = torch.tensor([i[3] for i in minibatch], dtype=torch.float32)
        # predicted_q_value_tensor = self.q_target(state_tensor)
        #
        # max_q_values = torch.max(predicted_q_value_tensor, 1)
        # max_q_values = max_q_values[0]
        # rewards = r + gamma*max_q_values
        return rewards

    def update_q_target_network(self):
        self.q_target.load_state_dict(self.q_network.state_dict())

class Loss_plots:
    #Initialise what is needed for the Plots
    average_losses = []
    variances = []
    iterations = []
    def __init__(self):
        # Create a graph which will show the loss as a function of the number of training iterations
        self.fig, self.ax = plt.subplots()

    def add_loss(self, loss, training_iteration):
        self.iterations.append(training_iteration)
        if loss == 0:
            self.average_losses.append(np.nan)
            return
        self.average_losses.append(loss)

    def plot(self):
        self.ax.set(xlabel='Episodes', ylabel='Loss', title='Loss Curve when using Replay Buffer')
        self.ax.set_yscale('log')
        self.ax.plot(self.iterations, self.average_losses, color='blue')
        print ("The variance of the loss in this graph is: ", np.nanvar(self.average_losses))
        return plt.show()
