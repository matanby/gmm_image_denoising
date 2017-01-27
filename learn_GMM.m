function [theta, LL] = learn_GMM(X, K, params0, options)
% Learn parameters for a gaussian mixture model via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   params0 - An optional struct with intialization parameters. Has 3
%             optional fields:
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK array, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector (should sum to 1).
%             If not given, a random starting point is generated.
%   options - Algorithm options struct, with fields:
%              learn - A struct of booleans, denoting which parameters
%                      should be learned: learn.means, learn.covs and
%                      learn.mix. The default is that given parameters
%                      (in params0) are not learned.
%              max_iter - maximum #iterations. Default = 1000.
%              thresh - if previous_LL * thresh > current_LL,
%                       algorithm halts. default = 1.0001.
%              verbosity - either 'none', 'iter' or 'plot'. default 'none'.
% Returns:
%   theta - A struct with learned parameters (fields):
%               means - a KxD matrix whose every row corresponds to a
%                       component mean.
%               covs - A DxDxK array, whose every page is a component
%                      covariance matrix.
%               mix - A Kx1 mixture vector.
%   LL - log likelihood history
%

if ~exist('params0', 'var') params0 = struct(); end
[theta, default_learn] = get_params0(X, K, params0);

if ~exist('options', 'var') options = struct(); end
options = organize_options(options, default_learn);

if K == 1
    theta.means = 0;
    theta.mix = 1;
    theta.covs = cov(X', 1);
    LL = GMM_loglikelihood(X, theta);
else         
    [D, M] = size(X);
    LL = -inf(options.max_iter, 1);        
            
    % X_XT is an MxDxD matrix where X_XT(i,:,:) is the matrix  defined by X_i * X_i^T.
    X_XT = zeros(D, D, M);        
    for i=1:M
        X_XT(:,:,i) = X(:,i) * X(:,i)';
    end
    X_XT = log(X_XT);    
    
    %% EM loop
    for t = 2:options.max_iter   
        %% E-step   
        % G is an MxK matrix whos (i,k) cell is Pr(x_i | h=k).
        G = zeros(M, K);         
        for j=1:K
            G(:,j) = log_mvnpdf(X', theta.means(j,:), theta.covs(:,:,j));                
        end
        
        % A is an MxK matrix whos i'th row is alpha (theta.mix).
        A = log(repmat(theta.mix', M, 1));        
                
        weighted_G = G + A;                
        nom_denom_vec = logsum(weighted_G, 2);                                                
        
        %% M-step        
        % CALC COV
        if options.learn.covs                        
            for k=1:K                                             
                v1 = weighted_G(:,k);
                v2 = nom_denom_vec;
                v3 = v1 - v2;

                M1 = reshape(v3, 1, 1, M);
                M1 = repmat(M1, D);            
                M2 = X_XT + M1;
                M3 = logsum(M2, 3);                       
                theta.covs(:,:,k) = exp(M3 - logsum(v3));        
            end
        end
        
        % CALC ALPHA
        if options.learn.mix
            M1 = weighted_G;
            M2 = repmat(nom_denom_vec, 1, K);
            M3 = M1 - M2;
            alpha = exp(logsum(M3,1)') ./ M;
            theta.mix = alpha;
        end
        
        %% CALC LL
        LL(t) = GMM_loglikelihood(X, theta);
                
        %% Check for convergence
        if LL(t-1)*options.threshold > LL(t); 
            LL = LL(2:t);
            break; 
        end
    end
    
    if strcmp(options.verbosity, 'iter')
        fprintf('learn_GMM with K=%d finished after %d iterations. (LL=%.2f)\n', K, t, LL(t-1));    
    elseif strcmp(options.verbosity, 'plot')
        figure; plot(LL);
    end
    
end
end

function [params0, default_learn] = get_params0(X, K, params0)
% organizes the params0 struct and output the starting point of the
% algorithm - "params0".
default_learn.mix = false;
default_learn.means = false;
default_learn.covs = false;

[D,M] = size(X);

if ~isfield(params0, 'means')
    default_learn.means = true;
    params0.means = X(:,randi(M, [1,K]))';
    params0.means = params0.means + nanstd(X(:))*randn(size(params0.means));
end

if ~isfield(params0, 'covs')
    default_learn.covs = true;
    params0.covs = nan(D,D,K);
    for k = 1:K
        params0.covs(:,:,k) = nancov(X(:,randi(M, [1,10]))');            
    end
end

if ~isfield(params0, 'mix')
    default_learn.mix = true;
    params0.mix = rand(K,1);
    params0.mix = params0.mix / sum(params0.mix);
end

end

function [options] = organize_options(options, default_learn)
%organize the options.
if ~isfield(options, 'threshold') options.threshold = 1.0001; end
if ~isfield(options, 'max_iter') options.max_iter = 1000; end
if ~isfield(options, 'verbosity') options.verbosity = 'none'; end
if ~isfield(options, 'learn') options.learn = default_learn;
else
    if ~isfield(options.learn, 'means') options.learn.means = default_learn.means; end;
    if ~isfield(options.learn, 'covs') options.learn.covs = default_learn.covs; end;
    if ~isfield(options.learn, 'mix') options.learn.mix = default_learn.mix; end;
end
end