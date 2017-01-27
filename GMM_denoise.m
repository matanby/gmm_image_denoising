function [Xhat] = GMM_denoise(Y, gmm, noise)
% Denoises every column in Y, assuming a gaussian mixture model and white
% noise.
% 
% The model assumes that y = x + noise where x is generated from a GMM.
%
% Arguments
%  Y - A DxM matrix, whose every column corresponds to a patch in D
%      dimensions (typically D=64).
%  gmm - The mixture model, with 4 fields:
%          means - A KxD matrix where K is the number of components in
%                  mixture and D is the dimension of the data.
%          covs - A DxDxK array whose every page is a covariance matrix of
%                 the corresponding component.
%          mix - A Kx1 vector with mixing proportions.
%  noise - the std of the noise in Y.
%

[D, M] = size(Y);
[K, ~] = size(gmm.mix);

G = zeros(M, K); % G_ij = log_mvnpdf(y_i, theta_j) * mix_j)
for k=1:K
    means_k = gmm.means(k, :);
    covs_k = gmm.covs(:,:, k);
    mix_k = gmm.mix(k,:)';
    G(:, k) = log_mvnpdf(Y', means_k, covs_k) + log(mix_k * ones(M, 1));
end

Xhat = zeros(D,M);
for j=1:K
    covs_k = gmm.covs(:,:, j);    
    Xhat = Xhat + ((eye(D) + (covs_k \ (eye(D) * noise.^2) )) \ Y) .* repmat(exp(((G(:, j) - logsum(G, 2))))', D, 1);
end