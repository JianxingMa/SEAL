function [data, label] = node_graph2vector(pos, neg, A, K)
%  Usage: to convert nodes' enclosing subgraphs (both pos 
%         and neg) into real vectors
%  --Input--
%  -pos: indices of positive nodes
%  -neg: indices of negative nodes
%  -A: the observed graph's adjacency matrix from which to
%      to extract subgraph features
%  -K: the number of nodes in each node's subgraph
%  --Output--
%  -data: the constructed training data, each row is a 
%         nodes' enclosing subgraph vector representation
%  -label: a column vector of nodes' labels
%
%  *author: Muhan Zhang, Washington University in St. Louis

all = [pos; neg];
pos_size = size(pos, 1);
neg_size = size(neg, 1);
all_size = pos_size + neg_size;

% Generate labels
label = [ones(pos_size, 1); zeros(neg_size, 1)];

% Generate vector data
d = K * (K - 1) / 2;  % dim of data vectors
data = zeros(all_size, d);
one_tenth = floor(all_size / 10);
display('Subgraph Pattern Encoding Begins...')
tic;
%poolobj = parpool(feature('numcores'));  % uncomment this line and change for to parfor in next line to enable parallel computing
for i = 1: all_size
    ind = all(i, :);
    sample = subgraph2vector(ind, A, K);
    data(i, :) = sample;
    %i/all_size  % display finer progress
    progress = i / one_tenth;
    if ismember(progress, [1:10])
        display(sprintf('Subgraph Pattern Encoding Progress %d0%%...', progress));
    end
end
if exist('poolobj')
    delete(poolobj)
end
toc;
end


function sample = subgraph2vector(ind, A, K)
%  Usage: 1) to extract the enclosing subgraph for a node
%         2) to impose a vertex ordering for the vertices
%            of the enclosing subgraph using graph labeling
%         3) to construct an adjacency matrix and output
%            the reshaped vector
%
%  *author: Muhan Zhang, Washington University in St. Louis

D = K * (K - 1) / 2;  % the length of output vector

% Extract a subgraph of K nodes
nodes = [ind];
links_dist = [0];  % the graph distance to the initial node
dist = 0;
fringe = [ind];
nodes_dist = [0];
while 1
    dist = dist + 1;
    fringe = neighbors(fringe, A);
    fringe = setdiff(fringe, nodes, 'rows');
    if isempty(fringe)  % no more new neighbors, add dummy nodes
        subgraph = A(nodes, nodes);
        break
    end
    nodes = [nodes; fringe];
    nodes_dist = [nodes_dist; ones(length(fringe), 1) * dist];
    if size(nodes, 1) >= K  % nodes enough, extract subgraph
        subgraph = A(nodes, nodes);  % the unweighted subgraph
        break
    end
end

links_dist_matrix = bsxfun(@min, nodes_dist, nodes_dist');
lweight_subgraph = subgraph ./ links_dist_matrix;

% Calculate the graph labeling of the subgraph
order = g_label(subgraph, nodes_dist);
if length(order) > K  % if size > K, delete the last size-K vertices and reorder
    order(K + 1: end) = [];
    subgraph = subgraph(order, order);
    lweight_subgraph = lweight_subgraph(order, order);
    order = g_label(subgraph);
end

% Generate enclosing subgraph's vector representation
ng2v = 2;  % method for transforming a g_labeled subgraph to vector 
switch ng2v
    case 1  % the simplest way -- one dimensional vector by ravelling adjacency matrix
        psubgraph = subgraph(order, order);  % g_labeled subgraph
        sample = psubgraph(triu(logical(ones(size(subgraph))), 1));
        sample(1) = sample(1) + eps;
    case 2  % use link distance-weighted adjcency matrix, performanc is better
        plweight_subgraph = lweight_subgraph(order, order);  % g_labeled link-weighted subgraph
        sample = plweight_subgraph(triu(logical(ones(size(subgraph))), 1));
        sample(1) = sample(1) + eps;  % avoid empty vector in libsvm format (empty vector results in libsvm format error)
end
if length(sample) < D  % add dummy nodes if not enough nodes extracted in subgraph
    sample = [sample; zeros(D - length(sample), 1)];
end
end


function N = neighbors(fringe, A);
%  Usage: find the neighbor nodse of all nodes in fringe from A

N = [];
for no = 1: size(fringe, 1)
    ind = fringe(no, :);
    [~, neis] = find(A(ind, :));
    N = [N; neis'];
    N = unique(N, 'rows', 'stable');  % eliminate repeated ones and keep in order
end
end


function order = g_label(subgraph, initial_colors)
%  Usage: impose a vertex order for a enclosing subgraph using graph labeling

p_mo = 7;  % use the palette_wl labeling

K = size(subgraph, 1);  % local variable

% switch different graph labeling methods
switch p_mo
case 1  % use classical wl, no initial colors
classes = wl_string_lexico(subgraph);
order = canon(full(subgraph), classes)';
case 2  % use wl_hashing, no initial colors
classes = wl_hashing(subgraph);
order = canon(full(subgraph), classes)';
case 3  % use classical wl, with initial colors
classes = wl_string_lexico(subgraph, initial_colors);
order = canon(full(subgraph), classes)';
case 4  % use wl_hashing, with initial colors
classes = wl_hashing(subgraph, initial_colors);
order = canon(full(subgraph), classes)';
case 5  % directly use nauty for canonical labeling
order = canon(full(subgraph), ones(K, 1))';
case 6  % no graph labeling, directly use the predefined order
order = [1: 1: K];
case 7  % palette_wl with initial colors, break ties by nauty
classes = palette_wl(subgraph, initial_colors);
%classes = palette_wl(subgraph);  % no initial colors
order = canon(full(subgraph), classes)';
case 8  % random labeling
order = randperm(K);
end
end
