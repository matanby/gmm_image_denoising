function N = lognormat(A, dim)
%LOGNORMAT normalize A along dim, in log space.
% Arguments:
%   A - matrix to normalize
%   dim - dimension along which to normalize. default is 2
%         (column)

if ~exist('dim','var') dim = 2; end;

rep_dims = ones(1,length(size(A))); rep_dims(dim) = size(A, dim);
N =  A - repmat(logsum(A,dim), rep_dims);
end