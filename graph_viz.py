import networkx as nx
import matplotlib.pyplot as plt
import threading
import time

class GraphViz(threading.Thread):
    def __init__(self, neurons):
        threading.Thread.__init__ (self)
        self.neurons = neurons

        self.G = nx.DiGraph()
        for neuron in neurons:
            self.G.add_node(neuron.id, potential=neuron.potential)
            for neighbour in neuron.axon_neighbours:
                self.G.add_edge(neuron.id,neighbour, weight=round(neuron.weights[neighbour],2))

    def run(self):
        pos=nx.spring_layout(self.G)
        while True:
            plt.clf()
            node_colors = []
            for node in self.G.nodes():
                node_colors.append('r' if round(self.neurons[node-1].potential)==1 else 'b')
            edge_labels = {(n1,n2): self.G[n1][n2]['weight'] for (n1,n2) in self.G.edges()}

            nx.draw_networkx(self.G, pos, node_size=700, node_color=node_colors)
            nx.draw_networkx_edge_labels(self.G , pos, edge_labels=edge_labels)
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.2)
        plt.show()
