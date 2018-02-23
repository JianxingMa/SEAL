SEAL -- learning from Subgraphs, Embeddings, and Attributes for Link prediction
===============================================================================

About
-----

Code for SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction). SEAL is a novel framework for link prediction which systematically transforms link prediction to a subgraph classification problem. For each target link, SEAL extracts its *h*-hop enclosing subgraph *A* and builds its node information matrix *X* (containing latent embeddings and explicit attributes of nodes). Then, SEAL feeds *(A, X)* into a graph neural network (GNN) to classify the link existence, so that it can learn from both graph structure features (from *A*) and latent/explicit features (from *X*) simultaneously for link prediction.

How to run
----------

Please download our [\[DGCNN software\]](https://github.com/muhanzhang/DGCNN) to the same level as this SEAL folder. DGCNN is the default graph neural network in SEAL.

Install the required libraries of DGCNN. Then run "Main.m" in MATLAB to do the experiments.

To be polished.

in SEAL, I assumed gpu number = 4.
change DGCNN_path in SEAL.m
change liblinear's matlab names

modifies the evaluation code of baseline methods, support more strict evaluation (all methods use the same positive and negative testing links to evaluate AUC)

Now the main time bottleneck is the file I/O between MATLAB and Torch. To let Torch read .mat graphs, I have to first save .mat to disk, then convert .mat to .dat by Torch, then save .dat to disk, and finally load .dat by DGCNN. The I/O took too much time.


Muhan Zhang, Washington University in St. Louis
2/10/2018
