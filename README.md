SEAL -- learning from Subgraphs, Embeddings, and Attributes for Link prediction
===============================================================================

About
-----

Code for SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction).

Please also download our DGCNN software to the same level as this folder.

To be polished.

in SEAL, I assumed gpu number = 4.
change DGCNN_path in GNN.m
parelle in Main and parellel in GNN.m are not compatible. Change it back to GNN-parallel when released.
change liblinear's matlab names

modifies the evaluation code of baseline methods, support more strict evaluation (all methods use the same positive and negative testing links to evaluate AUC)

Now the main time bottleneck is the file I/O between MATLAB and Torch. To let Torch read .mat graphs, I have to first save .mat to disk, then convert .mat to .dat by Torch, then save .dat to disk, and finally load .dat by DGCNN. The I/O took too much time.


Muhan Zhang, Washington University in St. Louis
2/10/2018
