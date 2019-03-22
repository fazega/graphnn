import threading
import time
import numpy as np
import utils

import config

class Neuron(threading.Thread):
    def __init__(self, id, axon_neighbours, dendrite_neighbours, message_queue, message_queue_loss):
        threading.Thread.__init__ (self)
        self.id = id
        self.message_queue = message_queue
        self.message_queue_loss= message_queue_loss
        self.axon_neighbours = axon_neighbours
        self.dendrite_neighbours = dendrite_neighbours

        self.potential = 0

        a = np.random.random(len(axon_neighbours))
        # a /= a.sum()
        self.weights = {}
        for i in range(len(self.axon_neighbours)):
            self.weights[self.axon_neighbours[i]] = a[i]
        print("Weights for neuron "+str(self.id)+" are "+str(self.weights))

        self.backpropT = threading.Thread(target=self.backprop)
        self.propT = threading.Thread(target=self.prop)

    def backprop(self):
        while True:
            time.sleep(config.window_backprop)
            for (neuron_id, error_grad) in self.message_queue_loss[self.id].copy():
                if(neuron_id != 0):
                    self.weights[neuron_id] -= config.nu*error_grad*self.potential
                    for neighbour in self.dendrite_neighbours:
                        x = config.grad_discount*(config.nu*error_grad*utils.gradSigmoidActivation(self.potential)*self.weights[neuron_id])
                        # print(x)
                        if(abs(x) > config.abs_grad):
                            self.message_queue_loss[neighbour].append((self.id, x))
                else:
                    for neighbour in self.dendrite_neighbours:
                        x = (self.potential-error_grad)*utils.gradSigmoidActivation(self.potential)
                        self.message_queue_loss[neighbour].append((self.id, x))
            self.message_queue_loss[self.id] = []

    def prop(self):
        while True:
            time.sleep(config.window_prop)
            input_neuron = False
            for (neuron_id, value) in self.message_queue[self.id].copy():
                self.potential += value
                if(neuron_id == 0):
                    input_neuron = True
                    self.potential = value
                    break
            if(not input_neuron):
                self.potential = utils.sigmoidActivation(self.potential)
            # print(len(self.message_queue_loss[self.id]))

            self.message_queue[self.id] = []
            for neighbour in self.axon_neighbours:
                self.message_queue[neighbour].append((self.id, self.potential*self.weights[neighbour]))


    def run(self):
        self.backpropT.start()
        self.propT.start()
