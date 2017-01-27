function [ps] = sample_patches(ims, psize, N, rmmean)
% sample N psized patches from ims uniformly
%
% Arguments:
%   ims - a cell array containing images.
%   psize - the size of the patch, if scalar assumed to be square
%           (default = 8)
%   N - number of total patches to sample. default = 1e4.
%   rmmean - whether mean should be removed. default = true
%
% Returns:
%   ps - N psized patches.
%
if ~exist('psize','var') || isempty(psize) psize = 8; end;
if ~exist('N','var') || isempty(N) N = 1e3; end;
if ~exist('rmmean','var') || isempty(rmmean) rmmean = true; end;
if isscalar(psize) psize = [psize, psize]; end;

P = numel(nan(psize));
nim = length(ims);
sizes = cellfun(@(x)numel(x),ims);

if rmmean
    mu = sum(cellfun(@(x)sum(x(:)),ims))./sum(sizes);
end
to_sample = ceil(sizes .* N / sum(sizes)); %how many patches to sample from each image
while sum(to_sample) > N
    i = randi(nim,1);
    to_sample(i) = to_sample(i) - 1;
end

ps = nan(P, N);
frm = [0,cumsum(to_sample(1:end-1)),N];
for i = 1:nim
    %random indices in current image
    idx = nan(P,to_sample(i));
    row_idx = randi(size(ims{i},1)-psize(1),[1,to_sample(i)]);
    col_idx = randi(size(ims{i},2)-psize(2),[1,to_sample(i)]);
    idx(1,:) = sub2ind(size(ims{i}),row_idx,col_idx)';
    for j = 2:psize(1)
        idx(j,:) = idx(j-1,:) + 1;
    end
    for j = 2:psize(2)
        idx((j-1)*psize+1:psize(1)*j,:) = idx(1:psize(1),:) + j * size(ims{i},1);
    end
     
    %add to patches
    ps(:,frm(i)+1:frm(i+1)) = reshape(ims{i}(idx(:)), [P, to_sample(i)]);
end

if rmmean
    ps = ps - mu;
end
end

