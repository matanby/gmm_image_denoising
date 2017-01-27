function [xhat] = denoise(y, model, noisestd, psize)
% denoise image y.
%
% This function splits y to subims in order to keep memory requirements
% down. To tweak subims size see MAX_SIZE variable below.
%
% Arguments:
%   y - a noisy gray-scale image.
%   model - a prior model with a "denoise" function handle field.
%   psize - the size of a patch use in model, if scalar interpreted as
%           square.
%   noisestd - the std of the noise.
%
if isscalar(psize) psize = [psize, psize]; end;

MAX_SIZE = [300,300];
xhat = nan(size(y));

%split to smaller images for memory considerations
[subims, rects] = imsplit(y, MAX_SIZE, psize); 
[K,L] = size(subims);
pind = sub2ind(psize, ceil(psize(1)/2), ceil(psize(2)/2)); %middle index in patch
for k = 1:K
    for l = 1:L
        sub = subims{k,l};
        rect = rects{k,l};
        noisy_patches = im2col(sub, psize);
        if ~all(size(noisy_patches) > 0) continue, end
        innerh = rect(3)-psize(1)+1;
        innerw = rect(4)-psize(2)+1;
        if innerh > 0 && innerw > 0
            patches = model.denoise(noisy_patches, noisestd);
            cleaned = reshape(patches(pind,:),[innerh, innerw]);
            fr_i = rect(1) + ceil(psize(1)/2) - 1;
            to_i = rect(1) + innerh + 2;
            fr_j = rect(2) + ceil(psize(2)/2) - 1;
            to_j = rect(2) + innerw + 2;
            xhat(fr_i:to_i,fr_j:to_j) = cleaned;
        end
    end
end
end