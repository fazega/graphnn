from neuron import Neuron
import time
import random
import utils
import threading


from graph_viz import GraphViz


n_neurons = 30
message_queue = {}
message_queue_loss = {}
for i in range(1,n_neurons+1):
    message_queue[i] = []
    message_queue_loss[i] = []

neurons = []
# neurons.append(Neuron(0, [1], [], message_queue,message_queue_loss, input_neuron=False))
# neurons.append(Neuron(1, [2], [], message_queue,message_queue_loss))
# neurons.append(Neuron(2, [], [1],message_queue,message_queue_loss))

# neurons.append(Neuron(1, [2,4,10], [2,4,5,8,10], message_queue,message_queue_loss))
# neurons.append(Neuron(2, [6,2,1], [1,3,6],message_queue,message_queue_loss))
# neurons.append(Neuron(3, [2,4,5], [1,2,6],message_queue,message_queue_loss))
# neurons.append(Neuron(4, [1,5], [1,2,3,5],message_queue,message_queue_loss))
# neurons.append(Neuron(5, [1,4,6], [3,4],message_queue,message_queue_loss))
# neurons.append(Neuron(6, [3,2], [5],message_queue,message_queue_loss))
# neurons.append(Neuron(7, [8,9], [5],message_queue,message_queue_loss))
# neurons.append(Neuron(8, [1,4,10], [5],message_queue,message_queue_loss))
# neurons.append(Neuron(9, [3,7,8,2], [5],message_queue,message_queue_loss))
# neurons.append(Neuron(10, [1,5,6,7], [5],message_queue,message_queue_loss))

for i in range(1,n_neurons+1):
    n = random.randrange(n_neurons//2)
    axon_neighbours = random.sample(range(1,n_neurons+1), n)
    neurons.append(Neuron(i, axon_neighbours, [], message_queue, message_queue_loss))

viz = GraphViz(neurons)
viz.start()


for neuron in neurons:
    neuron.start()


while True:
    x = 2*random.random()-1
    message_queue[1].append((0,x))
    print("Energy inserted ! Value "+str(x))
    time.sleep(2)
    print("State : "+str([neuron.potential for neuron in neurons]))


while True:
    x = random.random()
    f = lambda x: 0 if x < 0.5 else 1
    print("Putting "+str(x)+" in the network.")
    message_queue[1].append((0,x))
    time.sleep(0.1)
    t0 = time.time()
    fired = 0
    while time.time()-t0 < 0.2:
        if(round(neurons[1].potential) == 1):
            fired = 1
    message_queue_loss[2].append((0,abs(fired-f(x))))
    # print("New loss append.")
    # print("Neuron 2 is activated : "+str(round(neurons[1].potential)))
    # print("Neuron 2 potential : "+str(neurons[1].potential))
    # print("Weights for neuron 2 : "+str(neurons[1].weights))
    # print("Neuron 1 has potential "+str(neurons[0].potential))
    # print("Weights for neuron 1 : "+str(neurons[0].weights))
    print("Error : "+str(abs(fired-f(x))))
    print("\n")
