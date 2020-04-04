from nb_classifier import NaiveBayerClassifier
from multiprocessing import Pool
from itertools import product
import json

def findHighestFMeasureByWeight(i, divisor, classifier, newN, trainingFileName):
    weight = i/divisor
    currentStat = classifier.runML(newN= newN, newWeight=weight, fileName=trainingFileName)
    return [currentStat["f_measure"], weight, currentStat["weighted_f_measure"]]


class HyperParameterOptimizer:
# Perform on discrete value sweep using different hyperparameter to optimize the ML model

    MAX_N = 3
    MAX_WEIGHT = 1
    MAX_V = 4

    EPSILON = 0.0001

    def __init__(self, *args, **kwargs):
        self.VtoClassifer = {}

        self.stats = {}

        for i in range(1, self.MAX_V + 1):
            self.VtoClassifer[i] = NaiveBayerClassifier(N=self.MAX_N, V=i, weight=self.MAX_WEIGHT, test=False)
        
    
    def trainAllClassifiers(self, fileName):
        print("***** Training initial Models *****")
        for i in self.VtoClassifer:
            self.VtoClassifer[i].trainFromTweets(fileName)
            print("\t Models for V" + str(i) + " trained")

    def optimizeWeightForFMeasure(self, classifier, trainingFileName, newN, useWeightedF_M = False, indentation= "\t\t"):
        # Perform a weight sweep from 1->0. Stops when F-Measure converges to a final value (i.e. delta(F1-F2) < Epsilon, where Epsilon is a small error constant)
        stats = {"V": classifier.V, "N": newN}
        startWeight = 1.0
        f_delta = 1.0
        divisor = 10.0
        res = []
        print(indentation + "**** Finding optimal weight for " + str(stats))
        pool = Pool()

        while f_delta > self.EPSILON:
            multithreadInput = []
            for i in range(10, 0, -1):
                newInput = [i]
                newInput.append(divisor)
                newInput.append(classifier)
                newInput.append(newN)
                newInput.append(trainingFileName)
                multithreadInput.append(newInput)
            print("\t" + indentation + "*** " + str(stats) + " Range [" + str(10/divisor) + ", " + str(1/divisor) + "]")
            print("\t\t" + indentation + "*** Running multiprocess pool...")
            result = pool.starmap(findHighestFMeasureByWeight, multithreadInput)
            res.extend(result)
            '''
            for i in range(10, 0, -1):
                weight = i/divisor
                currentStat = classifier.runML(newWeight=weight, fileName=trainingFileName)
                if useWeightedF_M:
                    res.append([currentStat["weighted_f_measure"], weight])
                else:
                    res.append([currentStat["f_measure"], weight])
            '''
            if useWeightedF_M:
                f_delta = abs(res[-2][2] - res[-1][2])
            else:
                f_delta = abs(res[-2][0] - res[-1][0])
            divisor *= 10
        if useWeightedF_M:
            res.sort(key= lambda x: x[2], reverse = True)
        else:
            res.sort(key= lambda x: x[0], reverse = True)

        #print(res)
        print(indentation+ "** Optimal Weight for N=" + str(newN) + " V=" + str(classifier.V) +": ")
        print(indentation + str(res[0][1]))
        print(indentation +"Best F-measure: ")
        print(indentation + str(res[0][0]))
        print(indentation +"Best Weighted F-measure: ")
        print(indentation + str(res[0][2]))
        print(indentation +"** See log for complete dataset")

        if "V=" + str(classifier.V) not in self.stats:
            self.stats["V=" + str(classifier.V)] = {}

        if "N=" + str(newN) not in self.stats["V=" + str(classifier.V)]:
            self.stats["V=" + str(classifier.V)]["N=" + str(newN)] = {}
        
        self.stats["V=" + str(classifier.V)]["N=" + str(newN)] = {"Optimal Weight=": res[0][1], "Best F_Measure=": res[0][0], "Best Weighted F Measure=": res[0][2]}
        return res[0]
        

    def optimizeNandWeight(self, classifier, trainingFileName, useWeightedF_M=False, indentation= "\t"):
        stats = {"V": classifier.V}
        print(indentation + "**** Finding optimal N for " + str(stats))

        res = []
        for N in range(1, self.MAX_N+1):
            data = self.optimizeWeightForFMeasure(classifier, trainingFileName, newN=N, useWeightedF_M=useWeightedF_M)
            data.append(N)
            res.append(data)

        if useWeightedF_M:
            res.sort(key= lambda x: x[2], reverse = True)
        else:
            res.sort(key= lambda x: x[0], reverse = True)

        print(indentation+ "** Optimal N for V=" + str(classifier.V) +": ")
        print(indentation + str(res[0][3]))
        print(indentation +"Best F-measure: ")
        print(indentation + str(res[0][0]))
        print(indentation +"Best Weighted F-measure: ")
        print(indentation + str(res[0][2]))
        print(indentation +"Achieved with weight: ")
        print(indentation + str(res[0][1]))

        self.stats["V=" + str(classifier.V)]["Optimal N"] = {"Optimal N=": res[0][3], "Best F_Measure=": res[0][0], "Best Weighted F Measure=": res[0][2], "Best Setup Weight": res[0][1]}
        return res[0]

    def optimizeVandNandWeight(self, trainingFileName, useWeightedF_M=False):
        print( "**** Finding optimal V, N, Weight:")
        res = []
        for V in self.VtoClassifer:
            data = self.optimizeNandWeight(self.VtoClassifer[V], trainingFileName, useWeightedF_M)
            data.append(V)
            res.append(data)

        if useWeightedF_M:
            res.sort(key= lambda x: x[2], reverse = True)
        else:
            res.sort(key= lambda x: x[0], reverse = True)

        print("** Optimal V")
        print(res[0][4])
        print("** Optimal N")
        print(str(res[0][3]))
        print("** Optimal weight: ")
        print(str(res[0][1]))
        print("Best Possible F-measure: ")
        print(str(res[0][0]))
        print("Best Possible Weighted F-measure: ")
        print(str(res[0][2]))
        
        self.stats["Optimal Hyperparamenter"] = {"Optimal V":res[0][4], "Optimal N": res[0][3], "Optimal Weight": res[0][1], "Best F-measure": res[0][0], "Best Weighted F-measure": res[0][2]}
        self.printStats()

        return res[0]


    def printStats(self):
        print(json.dumps(self.stats, indent=4, sort_keys=True))

if __name__ == "__main__":
    HP_Optimizer = HyperParameterOptimizer()
    HP_Optimizer.trainAllClassifiers("training-tweets.txt")
    HP_Optimizer.optimizeVandNandWeight("test-tweets-given.txt", useWeightedF_M=True)