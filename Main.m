%  Main Program. Partly adapted from the codes of 
%  Lu 2011, Link prediction in complex networks: A survey.
%
%  *author: Muhan Zhang, Washington University in St. Louis

delete(gcp('nocreate'))

addpath(genpath('utils'));
datapath = 'data/';


setting = 1;
switch setting
case 1  % traditional link prediction benchmarks
    numOfExperiment = 5;        
    ratioTrain = 0.9;            
    connected = false; % whether to remove test links such that the remaining net is connected
    dataname = strvcat('USAir','NS','PB','Yeast','Celegans','Power','Router','Ecoli');
    dataname = strvcat('PB');
    dataname = strvcat('Yeast','Celegans','Power','Router');
    dataname = strvcat('Celegans', 'Power');
    dataname = strvcat('Yeast');
    %method = [1, 2, 3, 4, 5, 6, 7];  % 1: WLNM,  2: common-neighbor-based,  3: path-based, 4: random walk  5: latent-feature-based,  6: stochastic block model,  7: DGCNN,  8: WL graph kernel (out of memory on PB, Yeast and Ecoli), 9: embedding methods
    method =[8];
    h = 'auto';  % the maximum hop to extract enclosing subgraph
    include_embedding = 1;
    include_attribute = 0;
    portion = 1;  % portion of observed links selected as training data
case 2  % network embedding benchmark datasets
    % settings of node2vec experiments (for link prediction)
    numOfExperiment = 5;        
    ratioTrain = 0.5; 
    dataname = strvcat('facebook', 'arxiv');  % networks without node attributes
    dataname = strvcat('facebook', 'arxiv');  % networks without node attributes
    dataname = strvcat('arxiv');  % networks without node attributes
    connected = true;
    method = [1];
    h = 1;
    include_embedding = 1;
    include_attribute = 0;
    portion = 1;
case 3  % network embedding benchmark datasets
    % settings of node2vec experiments (for multi-label node classification)
    numOfExperiment = 5;        
    ratioTrain = 0.5; 
    dataname = strvcat('PPI_subgraph', 'Wikipedia', 'BlogCatalog');  % networks with node attributes
    dataname = strvcat('Wikipedia', 'PPI_subgraph');
    dataname = strvcat('BlogCatalog');
    dataname = strvcat('PPI_subgraph', 'Wikipedia', 'BlogCatalog');  % networks with node attributes
    connected = true;
    method = [8];
    h = 1;
    include_embedding = 1;
    include_attribute = 1;
    portion = 10000;  % randomly select 10000 observed links as positive training
end

tic;
num_in_each_method = [1, 13, 6, 13, 1, 1, 1, 1, 3];  % how many algorithms in each type of method
num_of_methods = sum(num_in_each_method(method));  % the total number of algorithms

auc_for_dataset = [];
for ith_data = 1:size(dataname, 1)                          
    tempcont = ['processing the ', int2str(ith_data), 'th dataset...', dataname(ith_data,:)];
    disp(tempcont);
    thisdatapath = strcat(datapath,dataname(ith_data,:),'.mat');  % load the net
    load(thisdatapath);                                 
    aucOfallPredictor = zeros(numOfExperiment, num_of_methods); 
    PredictorsName = [];
    
    % parallelize the repeated experiments
    %poolobj = parpool(feature('numcores')); % to enable it, uncomment this line and change 'for' to 'parfor' in the next line, note GNN and graph kernel don't support parallel experiments now (since they extract subgraphs parallelly inside)
    poolobj = parpool(numOfExperiment); 
    %poolobj = parpool(3); 
    parfor ith_experiment = 1:numOfExperiment
        ith_experiment
        if mod(ith_experiment, 10) == 0
                tempcont = strcat(int2str(ith_experiment),'%... ');
                disp(tempcont);
        end

        rng(ith_experiment);  % generate fixed network splits for different methods

        % divide into train/test
        [train, test] = DivideNet(net, ratioTrain, connected); % train test are now symmetric adjacency matrices without self loops                 
        train = sparse(train); test = sparse(test);  % convert to sparse matrices
        %train = spones(train + train'); test = spones(test + test');    

        % sample negative links for train and test sets (used by learning based methods, not by heuristic methods)
        htrain = triu(train, 1);  % half train adjacency matrix
        htest = triu(test, 1);
        [train_pos, train_neg, test_pos, test_neg] = sample_neg(htrain, htest, 1, portion);
        test = {};
        test.pos = test_pos; test.neg = test_neg; % evaluate performance on sampled test links
        train_mix = {};
        train_mix.train = train; % the observed network to extract enclosing subgraphs
        train_mix.pos = train_pos; train_mix.neg = train_neg;

        ithAUCvector = []; Predictors = []; % for recording results

        % run link prediction methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Weisfeiler-Lehman Neural Machine (WLNM)
        if ismember(1, method)
        disp('WLNM...');
        tempauc = WLNM(train_mix, test, 10, ith_experiment);                  % WLNM
            Predictors = [Predictors 'WLNM	'];      ithAUCvector = [ithAUCvector tempauc];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Common Neighbor-based methods, 13 methods
        if ismember(2, method)
        disp('CN...');
        tempauc = CN(train, test);                  % Common Neighbor
            Predictors = [Predictors 'CN	'];      ithAUCvector = [ithAUCvector tempauc];
        
        disp('Salton...');
        tempauc = Salton(train, test);              % Salton Index
             Predictors = [Predictors 'Salton	'];  ithAUCvector = [ithAUCvector tempauc];
        
        disp('Jaccard...');
        tempauc = Jaccard(train, test);             % Jaccard Index
             Predictors = [Predictors 'Jaccard	'];  ithAUCvector = [ithAUCvector tempauc];  

        disp('Sorenson...');
        tempauc = Sorenson(train, test);            % Sorenson Index
             Predictors = [Predictors 'Sorens	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('HPI...');
        tempauc = HPI(train, test);                 % Hub Promoted Index
             Predictors = [Predictors 'HPI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('HDI...');
        tempauc = HDI(train, test);                 % Hub Depressed Index
             Predictors = [Predictors 'HDI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHN...');
        tempauc = LHN(train, test);                 % Leicht-Holme-Newman
             Predictors = [Predictors 'LHN	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('AA...');
        tempauc = AA(train, test);                  % Adar-Adamic Index
             Predictors = [Predictors 'AA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RA...');
        tempauc = RA(train, test);                  % Resourse Allocation
             Predictors = [Predictors 'RA	'];       ithAUCvector = [ithAUCvector tempauc];  
       
        disp('PA...');
        tempauc = PA(train, test);                  % Preferential Attachment
             Predictors = [Predictors 'PA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LNBCN...');
        tempauc = LNBCN(train, test);               % Local naive bayes method - Common Neighbor
             Predictors = [Predictors 'LNBCN	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LNBAA...');
        tempauc = LNBAA(train, test);               % Local naive bayes method - Adar-Adamic Index
             Predictors = [Predictors 'LNBAA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LNBRA...');
        tempauc = LNBRA(train, test);               % Local naive bayes method - Resource Allocation
             Predictors = [Predictors 'LNBRA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Path-based methods, 6 methods
        
        if ismember(3, method)
        disp('LocalPath...');
        tempauc = LocalPath(train, test, 0.0001);   % Local Path Index
             Predictors = [Predictors 'LocalP	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('Katz 0.01...');
        tempauc = Katz(train, test, 0.01);          % Katz Index, beta=0.01
             Predictors = [Predictors 'Katz.01	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('Katz 0.001...');
        tempauc = Katz(train, test, 0.001);         % Katz Index, beta=0.001
             Predictors = [Predictors '~.001	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.9...');
        tempauc = LHNII(train, test, 0.9);          % Leicht-Holme-Newman II
             Predictors = [Predictors 'LHNII.9	'];    ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.95...');
        tempauc = LHNII(train, test, 0.95);         % Leicht-Holme-Newman II
             Predictors = [Predictors '~.95	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.99...');
        tempauc = LHNII(train, test, 0.99);         % Leicht-Holme-Newman II
             Predictors = [Predictors '~.99	'];       ithAUCvector = [ithAUCvector tempauc];  
             
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Random walk-based Methods, 13 methods

        if ismember(4, method)
        disp('ACT...');
        tempauc = ACT(train, test);                 % Average commute time
             Predictors = [Predictors 'ACT	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('CosPlus...');
        tempauc = CosPlus(train, test);             % Cos+ based on Laplacian matrix
             Predictors = [Predictors 'CosPlus	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RWR 0.85...');
        tempauc = RWR(train, test, 0.85);           % Random walk with restart (PageRank), d=0.85
             Predictors = [Predictors 'RWR.85	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RWR 0.7...');
        tempauc = RWR(train, test, 0.7);           % Random walk with restart, d=0.7
             Predictors = [Predictors '~.7	'];      ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SimRank 0.6...');
        tempauc = SimRank(train, test, 0.6);        % SimRank
             Predictors = [Predictors 'SimR	'];      ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 3...');
        tempauc = LRW(train, test, 3, 0.85);        % Local random walk, step 3
             Predictors = [Predictors 'LRW_3	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 4...');
        tempauc = LRW(train, test, 4, 0.85);        % Local random walk, step 4
             Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 5...');
        tempauc = LRW(train, test, 5, 0.85);        % Local random walk, step 5
             Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 3...');
        tempauc = SRW(train, test, 3, 0.85);        % Superposed random walk, step 3
             Predictors = [Predictors 'SRW_3	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 4...');
        tempauc = SRW(train, test, 4, 0.85);        % Superposed random walk, step 4
             Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 5...');
        tempauc = SRW(train, test, 5, 0.85);        % Superposed random walk, step 5
             Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];  
             
        disp('MFI...');
        tempauc = MFI(train, test);                 % Matrix forest Index
             Predictors = [Predictors 'MFI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('TS...');
        tempauc = TSCN(train, test, 0.01);          % Transfer similarity - Common Neighbor
             Predictors = [Predictors 'TSCN	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ismember(5, method)
        % latent feature models
        disp('MF...');
        tempauc = MF(train, test, 5, ith_experiment);                 % matrix factorization
             Predictors = [Predictors 'MF	'];       ithAUCvector = [ithAUCvector tempauc];  
        end
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ismember(6, method)
        % latent feature models
        disp('SBM...');
        tempauc = SBM(train, test, 12);                 % stochastic block models
             Predictors = [Predictors 'SBM	'];       ithAUCvector = [ithAUCvector tempauc];  
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Graph Neural Network for Link Prediction
        if ismember(7, method)
        disp('GNN...');
        tempauc = GNN(train_mix, test, h, ith_experiment, strip(dataname(ith_data, :)), include_embedding, include_attribute);
            Predictors = [Predictors 'GNN	'];      ithAUCvector = [ithAUCvector tempauc];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Graph Kernel + SVM for Link Prediction
        if ismember(8, method)
        disp('Kernel...');
        tempauc = graph_kernel(train_mix, test, h, ith_experiment, strip(dataname(ith_data, :)));
            Predictors = [Predictors 'Kernel	'];      ithAUCvector = [ithAUCvector tempauc];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Embedding + LR for Link Prediction
        if ismember(9, method)
        
        disp('Embedding...');
        tempauc = embedding_lp(train_mix, test, ith_experiment, strip(dataname(ith_data, :)), 'node2vec');
            Predictors = [Predictors 'node2vec	'];      ithAUCvector = [ithAUCvector tempauc];
        
        disp('Embedding...');
        tempauc = embedding_lp(train_mix, test, ith_experiment, strip(dataname(ith_data, :)), 'LINE');
            Predictors = [Predictors 'LINE	'];      ithAUCvector = [ithAUCvector tempauc];

        disp('Embedding...');
        tempauc = embedding_lp(train_mix, test, ith_experiment, strip(dataname(ith_data, :)), 'SPC');
            Predictors = [Predictors 'SPC	'];      ithAUCvector = [ithAUCvector tempauc];
        end


        aucOfallPredictor(ith_experiment, :) = ithAUCvector; PredictorsName = Predictors;
    end
    if exist('poolobj')
        delete(poolobj)
    end

    %% write the results for this dataset
    avg_auc = mean(aucOfallPredictor,1)
    auc_for_dataset = [auc_for_dataset, avg_auc];
    std_auc = std(aucOfallPredictor, 0, 1)
    respath = strcat(datapath,'result/',dataname(ith_data,:),'_res.txt');         
    dlmwrite(respath,{PredictorsName}, '');
    dlmwrite(respath,[avg_auc; std_auc], '-append','delimiter', '	','precision', 4);
    
end 
toc;
auc_for_dataset'


