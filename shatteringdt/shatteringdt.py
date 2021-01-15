import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeShattering:
    """This class is responsible to calculate the shattering coefficient for any
       sklearn decision tree classifier trained model and provide some tools to analyze some
       metrics that are useful to study a dataset to provide some theorical proofs of learning.
    """

    def __init__(self, tree):
        if(type(tree) == DecisionTreeClassifier):
            self.tree = tree
        else:
            raise "The Decision Tree model must be an sklearn.tree.DecisionTreeClassifier"

        # initializing variables
        self.delta = []
        self.samples = []
        self.shattering = []
        self.shat_confidence = []
        self.delta_confidence = []
        self.n_confidence_samples = []

        self._epsilon = .05
        self._n_samples = np.nan
        self._shattering_coefficient = np.nan

    def __repr__(self):
        return self.__class__

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n):
        self._n_samples = n

    @property
    def shattering_coefficient(self):
        return np.round(self._shattering_coefficient, 4)

    @shattering_coefficient.setter
    def shattering_coefficient(self, shat_coef_value):
        self._shattering_coefficient = shat_coef_value

    def g(self, n):
        """This function is responsible to apply the function g on the decision tree leaves in order to compute the shattering coefficient

        Parameters
        ----------
        n : float
            the number that corresponds to the proportion of the examples in a given node of decision tree.

        Returns
        -------
        float
            the function g applied.
        """
        return n*(self.n_samples+1.0)

    def recurse(self, left, right, node, parent, samples, shattering):
        """This function is responsible to realize the recursion over the decision tree in order to compute the shattering coefficient

        Parameters
        ----------
        left : sklearn node of the tree.
            the left child of the node
        right : sklearn node of the tree.
            the right child of the node
        node : sklearn node of the tree.
            the node to compute.
        parent : sklearn node of the tree.
            the father of the node.
        samples : list
            number of dataset samples in each node of the decisiont tree.
        shattering : list
            the shattering coefficient that are being computed.

        Returns
        -------
        float
            the computed shattering coefficient of the decision tree.
        """
        # check if the childrens are leave nodes
        if((left[left[node]] == -1) & (left[right[node]] == -1) & (right[right[node]] == -1) & (right[left[node]] == -1)):
            #print(f'Vou retornar um calculo de {samples[node]}/{samples[parent]}')
            return self.g(samples[node]/samples[parent])
        # if only left children is leaf node, returns 1
        elif((left[left[node]] == -1) & (right[left[node]] == -1)):
            return self.g(samples[node]/samples[parent])*(1 + self.recurse(left, right, right[node], node, samples, shattering))
        # if only right children is leaf node, returns 1
        elif((left[right[node]] == -1) & (right[right[node]] == -1)):
            return self.g(samples[node]/samples[parent])*(1 + self.recurse(left, right, left[node], node, samples, shattering))
        elif((left[node] != -1)) & (right[node] != -1):
            left_child = self.recurse(left, right, left[node], node, samples, shattering)
            right_child = self.recurse(left, right, right[node], node, samples, shattering)   
    
            if(parent is not None):
                return self.g(samples[node]/samples[parent])*(left_child + right_child)
            else:
                return self.g(1)*(left_child+right_child)

    def compute_shattering(self, prnt=True):
        """This function calculate the shattering_coefficient of the decision tree.

        Parameters
        ----------
        prnt : bool, optional
            Warn that the shattering coefficient is suscessfully computed, by default True
        """
        left = self.tree.tree_.children_left
        right = self.tree.tree_.children_right
        samples = self.tree.tree_.n_node_samples
        # calculating shattering coefficient.
        try:
            self.shattering_coefficient = self.recurse(left, right, 0, None, samples, [])
        except:
            raise "There's a problem computing shattering coefficient"
        
        if(prnt):
            print('Shattering coefficient has been computed')

    def get_computed_shattering(self):
        return f'The shattering coefficient computed for the provided tree with {self.n_samples} samples is {self.shattering_coefficient}'

    def simulate(self, epsilon=None, n_experimentations=200000):
        """This function realize simulations over the shattering coefficient to make measures to ensure learning guarentees in the model.

        Parameters
        ----------
        epsilon: float, optional
            the epsilon used in the confidence interval to ensure learning, by default .05 (class value)
        n_experimentations : int, optional
            number of experiments realized, by default 200000
        """
        self.delta = []
        self.samples = []
        self.shattering = []

        if(epsilon is not None):
            self.epsilon = epsilon

        # realizing experimentations
        for i in tqdm(range(1, n_experimentations)):
            self.n_samples = i
            self.samples.append(self.n_samples)
            self.compute_shattering(prnt=False)
            # calculating shattering
            self.shattering.append(self.shattering_coefficient)
            # calculating delta
            self.delta.append(2*self.shattering_coefficient*np.exp(-self.n_samples*(np.square(self.epsilon)/4)))

    def simulate_confidence_interval(self, max_confidence=0.20):
        """This function is responsible to simulate the number of samples that is needed to ensure learning given a confidence interval.

        Parameters
        ----------
        max_confidence : float, optional
            [description], by default 0.20
        """
        self.shat_confidence = []
        self.delta_confidence = []
        self.n_confidence_samples = []

        for epsilon in tqdm(np.arange(0.05, max_confidence+0.01, 0.01)):
            delta = 0
            self.n_samples = 1e7

            # simulation of a binary search to find fastly the number of samples.
            while(delta < epsilon):
                self.compute_shattering(prnt=False)
                delta = 2*self.shattering_coefficient*np.exp(-self.n_samples*(np.square(epsilon)/4))
                    
                if((self.n_samples % 2) == 0):
                    self.n_samples /= 2
                else:
                    self.n_samples = (self.n_samples+1)/2
                
            # after find a reasonal number of samples, the algorithm increment the number of samples until reach the exactly number of samples that match with the confidence interval epsilon.
            while(delta > epsilon):
                self.compute_shattering(prnt=False)
                delta = 2*self.shattering_coefficient*np.exp(-self.n_samples*(np.square(epsilon)/4))
                if(delta <= epsilon):
                    self.delta_confidence.append(epsilon)
                    self.n_confidence_samples.append(self.n_samples)
                    self.shat_confidence.append(self.shattering_coefficient)
                    
                self.n_samples += 1

    def plot_shattering(self, save=False):
        """This function plot the shattering coefficient versus number of samples. calculated in simulate function

        Parameters
        ----------
        save : bool, optional
            if true, save the image, by default False
        """
        if(self.shattering != []):
            fig = plt.figure(figsize=(18,10))
            plt.rcParams.update({'font.size': 16})

            plt.plot(self.shattering, linewidth=3)

            plt.xlabel('# of samples')
            plt.ylabel('Shattering Coefficient')
            plt.title('Measures of Shattering Coefficient by # of samples')

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

            plt.rcParams.update({'font.size': 10})
            plt.show()

            if(save):
                fig.savefig('shattering_coefficient.png', format='png', dpi=400)
        else:
            raise "You must run simulate() function in order to compute the values."

    def plot_chernoff_bound(self, save=False):
        if(self.shattering != []):
            fig = plt.figure(figsize=(18,10))
            plt.rcParams.update({'font.size': 16})

            plt.plot(self.delta, linewidth=3, alpha=2)

            plt.ylabel(r'$\delta$')
            plt.xlabel('# of samples')
            plt.title('Values of delta by # of samples')

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylim(0,1)
            plt.rcParams.update({'font.size': 10})

            plt.show()

            if(save):
                fig.savefig('chernoff_bound.png', format='png', dpi=400)
        else:
            raise "You must run simulate() function in order to compute the values."

    def plot_confidence_interval(self, save=False):
        """This function is responsible for plot the number of sampels needed to ensure learning for a given confidence interval.

        Parameters
        ----------
        save : bool, optional
            if true, save the image, by default False
        """
        if(self.n_confidence_samples != []):
            fig = plt.figure(figsize=(18,10))
            plt.rcParams.update({'font.size': 16})

            plt.plot(np.multiply(self.delta_confidence, 100).round().astype(int), self.n_confidence_samples, linewidth=3, marker='o')
            plt.gca().set_xticks(np.multiply(self.delta_confidence, 100).round().astype(int))
            plt.yscale('log')

            plt.ylabel('# of samples')
            plt.xlabel('Confidence interval (%)')
            plt.title('Number of samples needed to ensure learning given a confidence interval value.')

            plt.rcParams.update({'font.size': 12})
            for xy in zip(np.multiply(self.delta_confidence, 100).round().astype(int), self.n_confidence_samples):
                plt.gca().annotate(text=int(xy[1]), xy=xy, textcoords='data')

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.tight_layout()

            plt.show()

            if(save):
                fig.savefig('chernoff_bound_confidence_interval.png', format='png', dpi=400)
        else:
            raise "You must run simulate_confidence_interval() function in order to compute the values."