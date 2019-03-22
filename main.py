from neuron import Neuron
import time
import random

n_neurons = 6
message_queue = {}
message_queue_loss = {}
for i in range(1,n_neurons+1):
    message_queue[i] = []
    message_queue_loss[i] = []

neurons = []
neurons.append(Neuron(1, [2,3,4], [4,5], message_queue,message_queue_loss))
neurons.append(Neuron(2, [3,4], [1,3,6],message_queue,message_queue_loss))
neurons.append(Neuron(3, [2,4,5], [1,2,6],message_queue,message_queue_loss))
neurons.append(Neuron(4, [1,5], [1,2,3,5],message_queue,message_queue_loss))
neurons.append(Neuron(5, [1,4,6], [3,4],message_queue,message_queue_loss))
neurons.append(Neuron(6, [3,2], [5],message_queue,message_queue_loss))

for neuron in neurons:
    neuron.start()

# Ajout d'input (énergie)
message_queue[1].append((0,3))
time.sleep(1)

while True:
    x = random.random()
    f = lambda x: 0 if x < 0.5 else 1
    message_queue[1].append((0,x))
    time.sleep(0.5)
    message_queue_loss[5].append((0,f(x)))
    print("New loss append.")
    print("Neuron 5 has potential "+str(neurons[4].potential))
    print("Neuron 1 has potential "+str(neurons[0].potential))
    print("Error : "+str(abs(neurons[4].potential-f(x))))
