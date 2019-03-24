import threading
import time
import numpy as np
import utils

import config

class Neuron(threading.Thread):
    def __init__(self, id, axon_neighbours, dendrite_neighbours, message_queue, message_queue_loss, input_neuron=False):
        threading.Thread.__init__ (self)
        self.id = id
        self.message_queue = message_queue
        self.message_queue_loss= message_queue_loss
        self.axon_neighbours = axon_neighbours
        self.dendrite_neighbours = dendrite_neighbours

        self.input_neuron = input_neuron

        self.potential = 0

        if(not self.input_neuron):
            a = 2*np.random.random(len(axon_neighbours))-1
        else:
            a = np.ones(len(axon_neighbours))
            a /= len(a)

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
                # print(self.id,neuron_id,error_grad)
                if(neuron_id != 0):
                    self.weights[neuron_id] -= config.nu*error_grad*self.potential
                    for neighbour in self.dendrite_neighbours:
                        x = config.grad_discount*(config.nu*error_grad*utils.gradSigmoidActivation(self.potential)*self.weights[neuron_id])
                        if(abs(x) > config.abs_grad):
                            self.message_queue_loss[neighbour].append((self.id, x))
                else:
                    for neighbour in self.dendrite_neighbours:
                        x = error_grad*utils.gradSigmoidActivation(self.potential)
                        self.message_queue_loss[neighbour].append((self.id, x))
            self.message_queue_loss[self.id] = []

    def prop(self):
        while True:
            time.sleep(config.window_prop)
            received_input = False
            self.potential = self.potential/2
            for (neuron_id, value) in self.message_queue[self.id].copy():
                received_input = True
                self.potential += value
            if(received_input):
                self.potential = utils.sigmoidActivation(self.potential)

            self.message_queue[self.id] = []
            for neighbour in self.axon_neighbours:
                self.message_queue[neighbour].append((self.id, self.potential*self.weights[neighbour]))


    def run(self):
        self.backpropT.start()
        self.propT.start()
