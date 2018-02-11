%  Main Program for node classification.
%
%  *author: Muhan Zhang, Washington University in St. Louis

addpath(genpath('utils'));
datapath = 'data/';

numOfExperiment = 1;        

setting = 2;
switch setting
case 1  % traditional link prediction benchmarks
    ratioTrain = 0.9;            
    connected = false; % whether to remove test links such that the remaining net is connected
    dataname = strvcat('USAir','NS','PB','Yeast','Celegans','Power','Router','Ecoli');
    %dataname = strvcat('PB','Yeast','Celegans','Power','Router','Ecoli');
    %dataname = strvcat('USAir');
    method = [1, 2, 3, 4, 5, 6, 7, 8];  % 1: WLNM,  2: common-neighbor-based,  3: path-based, 4: random walk  5: latent-feature-based,  6: stochastic block model,  7: DGCNN,  8: WL graph kernel
    %method =[7];
    h = 1;  % the maximum hop to extract enclosing subgraph
case 2  % network embedding benchmark datasets
    % settings of node2vec experiments
    dataname = strvcat('PPI_subgraph', 'Wikipedia');
    dataname = strvcat('BlogCatalog');
    dataname = strvcat('PPI_subgraph');
    ratioTrain = 0.5; 
    connected = true;
    method = [7];
    h = 2;
end

tic;
num_in_each_method = [1, 13, 6, 13, 1, 1, 1, 1];  % how many algorithms in each type of method
num_of_methods = sum(num_in_each_method(method));  % the total number of algorithms

auc_for_dataset = [];
for ith_data = 1:size(dataname, 1)                          
    tempcont = ['processing the ', int2str(ith_data), 'th dataset...', dataname(ith_data,:)];
    disp(tempcont);
    thisdatapath = strcat(datapath,dataname(ith_data,:),'.mat');    
    load(thisdatapath);                                 
    aucOfallPredictor = zeros(numOfExperiment, num_of_methods); 
    PredictorsName = [];
    
    % parallelize the repeated experiments
    %poolobj = parpool(feature('numcores')); % to enable it, uncomment this line and change 'for' to 'parfor' in the next line, note GNN and graph kernel don't support parallel experiments now (since they extract subgraphs parallelly inside)
    for ith_experiment = 1:numOfExperiment
        ith_experiment
        if mod(ith_experiment, 10) == 0
                tempcont = strcat(int2str(ith_experiment),'%... ');
                disp(tempcont);
        end
        data_name = strip(dataname(ith_data, :));
        data_name_i = [data_name, '_', num2str(ith_experiment)];

        rng(ith_experiment);  % generate fixed network splits for different methods
        
        % divide nodes into labeled and unlabeld
        N = size(net, 1);
        perm = randperm(N);
        train = perm(1: ceil(ratioTrain * N))';
        test = perm(ceil(ratioTrain * N) + 1: end)';

        % generate labels for labeled nodes (train) and unlabeled nodes (test)
        node_labels = group;
        nlabels = size(group, 2);
        test_labels = node_labels(test, :);
        train_labels = node_labels(train, :);
        node_labels(test, :) = zeros(size(test, 1), nlabels);  % set testing nodes' labels to the zero vector

        % load node embeddings and build the node information matrix
        node_embeddings = dlmread(['data/embedding/', data_name, '.emd']);
        node_embeddings(1, :) = [];
        tmp = node_embeddings(:, 1);
        node_embeddings(:, 1) = [];
        node_embeddings(tmp, :) = node_embeddings;

        %data = node_subgraph_extract(train, test, node_labels, network, h);  % use node_labels (unknown replaced by 0s)
        data = node_subgraph_extract(train, test, node_embeddings, net, h);  % use node embeddings
        label = uint8(full([train_labels; test_labels]));

        % save to tempdata/
        data_info = whos('data');
        label_info = whos('label');
        data_bytes = data_info.bytes + label_info.bytes;
        n_splits = ceil(data_bytes / 2e9);  % split data into < 2GB splits, so that they can be saved in default .mat (torch only reads default .mat and -v7.3 is not supported)
        split_size = ceil(size(label, 1) / n_splits);
        system(sprintf('mkdir tempdata/%s', data_name_i));
        system(sprintf('rm tempdata/%s/*', data_name_i));
        for i = 1: n_splits
            save_struct.(data_name) = data((i-1)*split_size+1: min(i*split_size, length(label)));
            save(['tempdata/', data_name_i, '/split_', num2str(i), '.mat'], '-struct', 'save_struct');
            save_struct2.(['l' lower(data_name)]) = label((i-1)*split_size+1: min(i*split_size, length(label)), :);
            save(['tempdata/', data_name_i, '/split_', num2str(i), '.mat'], '-struct', 'save_struct2', '-append');
        end
        clear data label

        % convert .mat to .dat format (for Torch neural network to read)
        system(sprintf('th generate_torch_graphs.lua -dataName %s -multiLabel -ith_experiment %d', data_name, ith_experiment));

        % run DGCNN
        DGCNN_path = '../DGCNN/';
        data_pos = ['tempdata/', data_name_i];
        %cmd = sprintf('th %smain.lua -dataName %s -nodeLabel original -multiLabel -nClass %d -inputChannel %d -save tempdata -testNumber %d -gpu 2 -maxEpoch 500 -k 0.6 -fixed_shuffle original', DGCNN_path, data_name, nlabels, nlabels, length(test))
        cmd = sprintf('th %smain.lua -dataPos %s -dataName %s -nodeLabel original -multiLabel -nClass %d -inputChannel %d -save tempdata -testNumber %d -gpu 1 -maxEpoch 500 -k 0.4 -fixed_shuffle original -noSortPooling', DGCNN_path, data_pos, data_name_i, nlabels, size(node_embeddings, 2), length(test))
        system(cmd);
        
        f1 = load(['tempdata/', data_name_i, '/finalF1'])

    end

    if exist('poolobj')
        delete(poolobj)
    end

    %respath = strcat(datapath,'result/',dataname(ith_data,:),'_res.txt');         
    %dlmwrite(respath,{PredictorsName}, '');
    %dlmwrite(respath,[avg_auc; var_auc], '-append','delimiter', '	','precision', 4);
    
end 
toc;


