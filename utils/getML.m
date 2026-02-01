function [M, L] = getML(Xs,Ys,Xt,Yt,options)

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% M      :  MMD matrix
%%%% L      :  Graph laplacian matrix (manifold)

    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end
    
    % Manifold feature learning
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
    Xt = double(Xt_new');

    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
 
    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian matrix, M
    if options.rho > 0
        kk=options.p;
        ng = size(X,2);
        if kk>=ng
            kk=ng-1;
        end    
        manifold.k = kk;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    %% Maximum mean discrepency matrix, M
    % Estimate mu
    mu = estimate_mu(Xs',Ys,Xt',Yt);
    % Construct MMD matrix
    e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
    M = e * e' * length(unique(Ys));
    N = 0;
    for c = reshape(unique(Ys),1,length(unique(Ys)))
        e = zeros(n + m,1);
        e(Ys == c) = 1 / length(find(Ys == c));
        e(n + find(Yt == c)) = -1 / length(find(Yt == c));
        e(isinf(e)) = 0;
        N = N + e * e';
    end
    M = (1 - mu) * M + mu * N;
    M = M / norm(M,'fro');

end