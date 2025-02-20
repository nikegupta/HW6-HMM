import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        num_hidden_states = self.transition_p.shape[1]
        num_observations = input_observation_states.shape[0]

        alphas = np.zeros((num_hidden_states,num_observations))
        for h_i in range(num_hidden_states):
            alphas[h_i,0] = self.prior_p[h_i] * self.emission_p[h_i][self.observation_states_dict[input_observation_states[0]]]
        
        # Step 2. Calculate probabilities
        for o_i in range(1,num_observations):
            for h_i in range(num_hidden_states):
                alphas[h_i,o_i] = np.sum(alphas[:,o_i-1] * self.transition_p[:, h_i]) * self.emission_p[h_i][self.observation_states_dict[input_observation_states[o_i]]]



        # Step 3. Return final probability 
        return np.sum(alphas[:,num_observations-1])
        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(self.hidden_states), len(decode_observation_states)))
        backpointer = np.zeros((len(self.hidden_states), len(decode_observation_states)))     
        
        # Step 2. Calculate Probabilities
        for o_i in range(len(decode_observation_states)):
            for h_i in range(len(self.hidden_states)):
                if o_i == 0:
                    viterbi_table[h_i][o_i] = self.prior_p[h_i] * self.emission_p[h_i][self.observation_states_dict[decode_observation_states[o_i]]]
                else:
                    max_prob = max(viterbi_table[prev_h_i][o_i-1] * self.transition_p[prev_h_i][h_i] for prev_h_i in range(len(self.hidden_states)))
                    viterbi_table[h_i][o_i] = max_prob * self.emission_p[h_i][self.observation_states_dict[decode_observation_states[o_i]]]
                    backpointer[h_i][o_i] = max(range(len(self.hidden_states)), key=lambda prev_h_i: viterbi_table[prev_h_i][o_i-1] * self.transition_p[prev_h_i][h_i])

        #Step 3. Traceback 
        best_path_pointer = max(range(len(self.hidden_states)), key=lambda h_i: viterbi_table[h_i][-1])
        best_path = [best_path_pointer]
        for i in range(len(decode_observation_states)-1,0,-1):
            best_path.append(backpointer[best_path[0]][i])

        best_path = best_path[::-1]
        best_hss = []
        for i in range(len(best_path)):
            best_hss.append(self.hidden_states[int(best_path[i])])

        # Step 4. Return best hidden state sequence 
        return best_hss
        