function [model] = learn_GSM(X, K, options)
% Learn parameters for a gaussian scaling mixture model for X via EM
%
% GSM components share the variance, up to a scaling factor, so we only
% need to learn scaling factors c_1.. c_K and mixture proportions
% alpha_1..alpha_K.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - a struct with 3 fields:
%           mix - Mixture proportions.
%           covs - A DxDxK array, whose every page is a scaled covariance
%                  matrix according to scaling parameters.
%           means - K 0-means.
%

if ~exist('options', 'var') 
    options = struct(); 
end

params0 = struct();
% In this model we assume all Gausians are of 0 mean.
params0.means = zeros(K, size(X, 1));

[D, ~] = size(X);
params0.covs = zeros(D, D, K);

cov_X = cov(X', 1);
for k = 1:K
    c_k = 2*k/K;
    params0.covs(:,:,k) = c_k * cov_X;
end

[model, LL] = learn_GMM(X, K, params0, options);
