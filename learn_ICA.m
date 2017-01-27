function model = learn_ICA(X, K, options)
% Learn parameters for a complete invertible ICA model.
%
% We learn a matrix P such that X = P*S, where S are D independent sources
% And for each of the D coordinates we learn a mixture of K (univariate)
% 0-mean gaussians via EM.
%
% Arguments:
%   X - Data, a DxM data matrix, where D is the dimension, and M is the
%       number of samples.
%   K - Number of components in a mixture.
%   options - options for learn_GMM (optional).
% Returns:
%   model - A struct with 3 fields:
%           P - mixing matrix of sources (P: D ind. sources -> D signals)
%           vars - a DxK matrix whose (d,k) element correponsds to the
%                  variance of the k'th component in dimension d.
%           mix - a DxK matrix whose (d,k) element correponsds to the
%                 mixing weight of the k'th component in dimension d.
%

if ~exist('options', 'var') 
    options = struct(); 
end

[D, ~] = size(X);

cov_X = cov(X', 1);
[P, ~] = eig(cov_X);
model.P = P;
S = model.P \ X;

model.vars = zeros(D, 1, 1, K);
model.mix = zeros(D, K);

params0 = struct();
% In this model we assume all Gausians are of 0 mean.
params0.means = zeros(K, 1);

for i=1:D
    [theta_i, LL] = learn_GMM(S(i,:), K, params0, options);    
    model.vars(i,:) = theta_i.covs;
    model.mix(i,:) = theta_i.mix';
end
