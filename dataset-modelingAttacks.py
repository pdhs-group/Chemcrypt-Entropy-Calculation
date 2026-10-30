from AttackModel import *
import sys
from utils import init

config = sys.argv[1] if len(sys.argv) > 1 else exit(1)

experiment = ExperimentConfiguration.from_directory(config)

layers = 4

model_per_layer = []


model_no_encoding = ModelExperiment.from_experimentConfiguration(experiment, group_by_print=True, layer_wise=False)

for i in range(1,layers+1):
    model = ModelExperiment.from_experimentConfiguration(experiment, group_by_print=True, layer_wise=True, exp_layer=i)
    model_per_layer.append(model)

KNN = KNNClassifier(use_pca=True)
NB = NaiveBayes_Classifier(use_pca=True)
LR = LogisticRegression_classifier(use_pca=True)
SVM = SVM_classifier(use_pca=True)

LR_noencoding = LogisticRegression_classifier(use_pca=True, max_iter=10000)
SVM_noencoding = SVM_classifier(use_pca=True, max_iter=10000)

#for model in model_per_layer:
    #attack_results_KNN = KNN.attack(model)
    #attack_results_KNN_cross_validate = KNN.cross_validate_attack(model)
    #print("---------------------------------------------")
    #attack_results_NB = NB.attack(model)
    #attack_results_NB_cross_validate = NB.cross_validate_attack(model)
    #print("---------------------------------------------")
    #attack_results_LR = LR.attack(model)
    #attack_results_LR_cross_validate = LR.cross_validate_attack(model)
    #print("###############################################")
    #attack_results_SVM = SVM.attack(model)
    #attack_results_SVM = SVM.cross_validate_attack(model)


attack_results_KNN_no_encoding = KNN.attack(model_no_encoding)
print("---------------------------------------------")
attack_results_NB_no_encoding = NB.attack(model_no_encoding)
print("---------------------------------------------")
attack_results_LR_no_encoding = LR_noencoding.attack(model_no_encoding)
print("---------------------------------------------")
attack_results_SVM_no_encoding = SVM_noencoding.attack(model_no_encoding)

