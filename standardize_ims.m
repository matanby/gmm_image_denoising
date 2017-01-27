function [sims] = standardize_ims(ims)
% converts images to greyscale, rescals to 0-1 and removes the mean pixel
% value of the entire dataset.

sims = cellfun(@(x)double(rgb2gray(x))/255, ims, 'uniformoutput', false);
mu = sum(cellfun(@(x)sum(x(:)), sims)) / sum(cellfun(@(x)numel(x), sims));
sims = cellfun(@(x)x-mu, sims, 'uniformoutput', false);

