function [subx, rects] = imsplit(X, max_size, overlap)
% split an image to subimages.

% Arguments:
%   X - the image.
%   max_size - either a scalar which is interpreted as a square, or a pair
%              [max_rows, max_cols]. This determines the maximal size
%              allowed for a resulting sub-image. default = half the image
%              size.
%   overlap - either a scalar which is interpreted as a square, or a pair
%             [row_overlap, col_overlap]. This determines the overlap in
%             rows and columns between subimages. default = 0.
% Returns:
%   subx - a cell array holding sub-images of X, in their order, so that
%          they cover X completely.
%   rects - a corresponding collection of tuples: (i, j, h, w) such that:
%           subx{k,l} = X(i:i+h-1,j:j+w-1)
%           rects is given as a 1x4 vector.
%

if ~exist('max_size','var') || isempty(max_size) max_size = ceil(size(X)/2); end;
if ~exist('overlap','var') || isempty(overlap) overlap = 0; end;
if isscalar(max_size) max_size = [max_size, max_size]; end;
if isscalar(overlap) overlap = [overlap, overlap]; end;

[M,N] = size(X);

sub_max_h = ceil(M / (max_size(1)-overlap(1)) );
sub_max_w = ceil(N / (max_size(2)-overlap(2)) );

Is = (1:floor(M/sub_max_h):M);
Js = (1:floor(N/sub_max_w):N);
rects = cell(length(Is),length(Js));
subx = cell(length(Is),length(Js));

for i_ind = 1:length(Is)
    i = Is(i_ind);
    h = min(i + max_size(1) - 1, M) - i + 1;
    for j_ind = 1:length(Js)
        j = Js(j_ind);
        w = min(j + max_size(2) - 1, N) - j + 1;
        subx{i_ind,j_ind} = X(i:i+h-1, j:j+w-1);
        rects{i_ind,j_ind} = [i,j,h,w];
    end
end
