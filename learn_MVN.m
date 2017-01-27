function [model] = learn_MVN(X, options)
% Learn parameters for a 0-mean multivariate normal model for X.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   options - options for learn_GMM (optional).
% Returns:
%   model - a struct with 3 fields:
%            cov - DxD covariance matrix.
%

if ~exist('options', 'var') 
    options = struct(); 
end

params0 = struct();
% In this model we assume all Gausians are of 0 mean.
params0.means = zeros(1, size(X, 1));

[theta, LL] = learn_GMM(X, 1, params0, options);
model.cov = theta.covs;
