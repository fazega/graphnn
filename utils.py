import config
import math

def reluActivation(potential):
    if(potential > config.bias_activation):
        return min(config.max_activation,potential)
    else:
        return 0

def sigmoidActivation(potential):
    return 1/(1+math.exp(-config.strength_activation*(potential-config.bias_activation)))

def gradSigmoidActivation(potential):
    return config.strength_activation*potential*(1-potential)
