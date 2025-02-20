import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    hmm = HiddenMarkovModel(mini_hmm['observation_states'],mini_hmm['hidden_states'],
                            mini_hmm['prior_p'],mini_hmm['transition_p'],
                            mini_hmm['emission_p'])

    result = hmm.forward(mini_input['observation_state_sequence'])
    best_hss = hmm.viterbi(mini_input['observation_state_sequence'])

    assert best_hss == list(mini_input['best_hidden_state_sequence'])
    assert result - 0.0350644 < 0.001

def test_full_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')
    hmm = HiddenMarkovModel(full_hmm['observation_states'],full_hmm['hidden_states'],
                            full_hmm['prior_p'],full_hmm['transition_p'],
                            full_hmm['emission_p'])

    result = hmm.forward(full_input['observation_state_sequence'])
    best_hss = hmm.viterbi(full_input['observation_state_sequence'])

    assert best_hss == list(full_input['best_hidden_state_sequence'])
    assert result == 1.6864513843961343e-11






    
   
    pass



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass













