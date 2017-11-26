import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object

        Implementation Description:
        we make a loop through all the possible given state numbers and for each one, we calculate the BIC value of them
        using BIC = -2* logL +p *logN where , logl is the likelihood, p is the number of free params that we have and N is
        the sum of the lenghts of the

        For calculating the free params I have used the following resources:

        From https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
        "If we develop the HMM using the GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
        random_state=self.random_state, verbose=False).fit(self.X, self.lengths) from hmmlearn we are calculating the
        following parameters that are the ones we use in BIC:
        Initial state occupation probabilities = numStates
        Transition probabilities = numStates*(numStates - 1)
        Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
        numMeans and numCovars are the number of means and covars calculated. One mean and covar for each state and
        features. Then the total number of parameters are:
        Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities

        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bICDict = {}
        for stateNum in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(stateNum)
                # Params =Initial state occupation probabilities + Transition probabilities + Emission probabilities
                params = stateNum + (stateNum * (stateNum - 1)) + (stateNum * sum(self.lengths) * 2)
                bICDict[model] = -2 * model.score(self.X, self.lengths) + params * math.log10(sum(self.lengths))
            except Exception as ex:
                pass
        return min(bICDict, key=bICDict.get)






class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    Implementation note:
    For this scoring algorithm, we need to first create the model based on the available sequences for the given word,
    then we need to reduce the likely hood of the word sequence from the likelihood of the all other sequences to find
    out which number of states could recognize most of the samples while giving the least possible number of false
    positive results ( recognizing other words mistakenly as the word under calculation)
    For achieving this, I have implemented two extra help functions, one for calculating the unlikelihood of other
    sequences against the created model and a second function to assemble a {word -> sequence, length} dictionary to be
    used inside the unlikelihood function.
    '''
    # implements a dictionary of all the X, lengths parameters for all the available words defined for the ModelSelector
    def get_x_length_dict(self):
        x_length_dict = {}
        for word in self.words:
            x_length_dict[word] = self.hwords[word]
        return x_length_dict

    # calculates the likelihood of other sequences being mistaken for the word under construction.
    # x_length_dict, is the dictionary that is implemented in get_x_length_dict function
    # model, is the model that is implemented for a certain number of states which is implemented in the select function
    def anti_likelihood(self, model, x_length_dict):
        anti_log = []
        # loop through all the available words for the ModelSelector
        for word in self.words:
            # ignore the word under construction
            if word is not self.this_word:
                word_x, word_lengths = x_length_dict.get(word)
                try:
                    anti_log.append(model.score(word_x, word_lengths))
                except Exception as ex:
                    pass
        # pass the average of likelihood values to the caller
        return sum(anti_log)/len(anti_log)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        x_length_dict = self.get_x_length_dict()
        dic_dictionary = {}
        # loop over possible state numbers
        for stateNum in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(stateNum)
                dic_dictionary[model] = model.score(self.X, self.lengths) - self.anti_likelihood(model, x_length_dict)
            except Exception as ex:
                pass
        # return the model with the highest score
        return max(dic_dictionary, key=dic_dictionary.get)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    Implementation description: For implementing the select functionality, we need to loop over all the possible number
    of states that we can have, and for each state if the number of available sequence samples are large enough, we
    break them down into three folds of test and training data set,
    for each fold then we train the HMM model with the training data and calculate the log likelihood of the test data
    and at the end we make an average on the log likelihood for each state length.
    Finally we compare the average log likelihoods of each state and return the HMM model of the state with the highest
    log likelihood average.
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # a dictionary where key= Hmm model result and value is the average log likelihood of cv
        averageLogDict = {}
        # loop over possible state numbers
        for stateNum in range(self.min_n_components, self.max_n_components):
            try:
                # is the sequence sample for the word large enough?
                if len(self.sequences) > 2:
                    split_method = KFold(n_splits=3, random_state=None, shuffle=False)
                    # placeholder for storing log-likelihood values for each fold
                    loglList = []
                    # retrieve a training set and a test set from the total sequences of a word
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        testX, testLengths = combine_sequences(cv_test_idx, self.sequences)
                        # training the model with the training set
                        model = self.base_model(stateNum)
                        loglList.append(model.score(testX, testLengths))
                    averageLogDict[model] = sum(loglList)/len(loglList)
                    # averageLogDict[model] = np.mean(loglList)
                # in case the sample length is not large enough, use the same sequences for finding the log likelihood
                # which was also being used for training the hmm model
                else:
                    model = self.base_model(stateNum)
                    averageLogDict[model] = model.score(self.X, self.lengths)
            # ignore the samples that are not compatible with the hmm implementer
            except Exception as ex:
                pass
        return max(averageLogDict, key=averageLogDict.get)
