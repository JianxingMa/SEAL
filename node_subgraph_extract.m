function [data] = node_subgraph_extract(train, test, node_information, A, h)
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

all = [train; test];
all_size = size(all, 1);

% Extract subgraphs
data = {};
one_tenth = floor(all_size / 10);
display('Subgraph Extraction Begins...')
tic;
%poolobj = parpool(3);
%poolobj = parpool(feature('numcores'));
for i = 1: all_size
    ind = all(i);
    sample = subgraph2mat(ind, node_information, A, h);

    data(i).am = sample.am;
    data(i).nl = sample.nl;

    %i/all_size  % display finer progress
    progress = i / one_tenth;
    if ismember(progress, [1:10])
        display(sprintf('Subgraph Extraction Progress %d0%%...', progress));
    end
end
if exist('poolobj')
    delete(poolobj)
end
toc;
end


function sample = subgraph2mat(ind, node_information, A, h)
%  Usage: 1) to extract the enclosing subgraph for a node
%         2) to impose a vertex ordering for the vertices
%            of the enclosing subgraph using graph labeling
%         3) to construct an adjacency matrix and output
%            the reshaped vector
%
%  *author: Muhan Zhang, Washington University in St. Louis

% Extract a subgraph of around the target node up to h hops
nodes = [ind];
links_dist = [0];  % the graph distance to the initial node
dist = 0;
fringe = [ind];
nodes_dist = [0];
for dist = 1: h
    fringe = neighbors(fringe, A);
    fringe = setdiff(fringe, nodes, 'rows');
    if isempty(fringe)  % no more new neighbors, add dummy nodes
        break
    end
    nodes = [nodes; fringe];
    nodes_dist = [nodes_dist; ones(length(fringe), 1) * dist];
    if dist == h  % nodes enough, extract subgraph
        break
    end
end
subgraph = A(nodes, nodes);
am = uint8(full(subgraph));  
al = cellfun(@(x) uint16(find(x)), num2cell(am, 2), 'un', 0);  % change to longer integer format if your graph size > 65535
% if using node embedding
nl = node_information(nodes, :);
% if using node labels
%labels(1, :) = zeros(1, size(labels, 2));  % we shouldn't include the target node's class information, thus set to unknown class
%nl = uint8(full(labels));  % node labels

for reorder = 1:0  % reorder the vertices using palette-wl graph labeling
    order = g_label(subgraph, nodes_dist + 1);
    am = am(order, order);
    al = al(order);
    nl = nl(order, :);
end

% build one sample
sample = {};
sample.am = am;
sample.al = al;
sample.nl = {};
sample.nl.values = nl;

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
K = size(subgraph, 1);  % local variable
classes = palette_wl(subgraph, initial_colors);
order = canon(full(subgraph), classes)';
end
