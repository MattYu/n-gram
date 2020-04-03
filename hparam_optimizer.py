from nb_classifier import NaiveBayerClassifier


class HyperParameterOptimizer:
# Perform on discrete value sweep using different hyperparameter to optimize the ML model

    def __init__(self, *args, **kwargs):
        self.VtoClassifer = {}

        for i in range(1, 5):
            self.VtoClassifer[i] = NaiveBayerClassifier(N=3, V=i, weight=0.000000001, test=False)
        
    
    def trainAllClassifiers(self, fileName):
        for i in self.VtoClassifer:
            self.VtoClassifer[i].trainFromTweets(fileName)

    def optimizeWeight(self, classifier):
        pass

    def optimizeNandWeight(self):
        pass

    def optimizeVandNandWeight(self):
        pass