from hmm import HiddenMarkovModel
import numpy as np

def main():
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(mini_hmm['observation_states'],mini_hmm['hidden_states'],
                            mini_hmm['prior_p'],mini_hmm['transition_p'],
                            mini_hmm['emission_p'])

    result = hmm.forward(mini_input['observation_state_sequence'])
    best_hss = hmm.viterbi(mini_input['observation_state_sequence'])
    print(result)
    print(best_hss)
    print(list(mini_input['best_hidden_state_sequence']))

    

if __name__ == "__main__":
    main()