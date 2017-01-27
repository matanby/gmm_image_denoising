function [N, S] = normat(A, dim)
%NORMATnormalize A along dim.
% Arguments:
%   A - matrix to normalize
%   dim - dimension along which to normalize. default is 2
%         (column)
%
% Returns:
%   N - the normalized (sums to 1 along dim) matrix.
%   S - the sums along dims.
%

if ~exist('dim','var') dim = 2; end;

rep_dims = ones(1,length(size(A))); rep_dims(dim) = size(A, dim);
S = nansum(A,dim);
N =  A ./ repmat(S, rep_dims);
end