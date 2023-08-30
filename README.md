# magic-QRW-with-CNN
Detecting quantum speedup for random walks with artificial neural networks

FILES:
graph_simulation.py
Simulate the random walk and quantum random walk on a graph.

corpus_generator.py
Generates a list of graphs: linear, random or circular, and gives the elements ranking with a call to graph_simulation.py.

make_dataset_from_corpus.py
Transforms the corpus, list with ranked graphs, to dataset that can be used for training the ANN.

QRWCNN.py
The architecture of the ANN.
