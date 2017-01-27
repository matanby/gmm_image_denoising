function [psnr, ll, dur] = test_denoising(test, models, noise_range,...
                                          to_plot, only_plot, psize)
% Arguments:
%  test - a cell array of images.
%  models - prior models. structs with fields:
%           * denoise - a function xhat <- y,noise
%           * loglikelihood - a function R <- x
%           * name - a string with the name of the model.
%  noise_range - a range of noise to be added to the picture.
%                default = [.001, .05, .1, .2].
%  to_plot - whether results should be plotted. default = [], i.e. don't.
%            If a pair [width, height] is given, it's interpreted as a
%            frame size to be plotted.
%  psize - patch size, either scalar (square ptch) or a pair. defaults to
%          8.
%  only_plot - if true, only the frames that are plotted are denoised
%              rather than complete images. usfeul for debugging.
% Returns:
%   psnr - a Ix(1+M)xN array with results where I=#images, M=#models,
%          N=#noise.
%   ll - a M lengthed vector of log likelihood of each model (estimated via 
%        10^5).
%   dur - the duration it took to denoise the images (a Ix(1+M)xN array);
%
% Usage:
%
% %learn ica and prep model struct
% >> ica_model = learn_ICA(train_x, 10, options);
% >> ica_model.name = 'ICA';
% >> ica_model.loglikelihood = @(x)ICA_loglikelihood(x, ica_model);
% >> ica_model.denoise = @(y, noise)ICA_denoise(y, ica_model, noise);
%
% %learn gsm and prep model struct
% >> gsm_model = learn_GSM(train_x, 20, options);
% >> gsm_model.name = 'GSM';
% >> gsm_model.loglikelihood = @(x)GMM_loglikelihood(x,gsm_model);
% >> gsm_model.denoise = @(y, noise)GMM_denoise(y,gsm_model,noise);
%
% % denoise, evaluate and plot
% >> [psnr, ll] = test_denoising(ims.test(2:3:8),{mvn_model, ica_model, gsm_model});
% ...
% >> bar(ll') % yay!

pSNR = @(x,y)-10*log10(nanmean((x(:)-y(:)).^2));

if ~exist('noise_range','var') || isempty(noise_range)
    noise_range = [.001, .05, .1];
end;
if ~exist('psize','var') || isempty(psize) psize = 8; end;
if ~exist('to_plot','var') || isempty(to_plot) to_plot = [100,160]; end;
if ~exist('only_plot','var') || isempty(only_plot) only_plot = true; end;

if isscalar(psize) psize = [psize, psize]; end;

%adding a model that does nothing to the mix
nomodel.denoise = @(y,noisestd) y;
nomodel.loglikelihood = @(x)0;
nomodel.name = 'noised';
models = {nomodel, models{:}};

if only_plot
    %keep only middle frame of images
    midframe = @(x) x((size(x,1) - to_plot(1))/2 - 1 : (size(x,1) + to_plot(1))/2 + 1,...
                      (size(x,2) - to_plot(2))/2 - 1: (size(x,2) + to_plot(2))/2 + 1);
    test = cellfun(midframe, test, 'uniformoutput', false);
end
    

I = length(test);
M = length(models);
S = length(noise_range);

psnr = nan(I,M,S);
dur = nan(I,M,S);
ll = nan(M,1);

if ~isempty(to_plot)
    pR = M;
    pC = S;
    ahs = nan(I,pR,pC); %axes handles for all figures
    for i = 1:I
        figure(i);
        clf;
        colormap(gray);
        ahs(i,:,:) = reshape(tight_subplot(pR, pC, [.01,.01],...
                                           [.01, .05], [.05, .01]), [S, M])';
    end
end

tst_ps = sample_patches(test, psize, 1e5);
for i = 1:I %images
    x = test{i};
    for si = 1:S %noise
        noise = noise_range(si);
        y = x + noise * randn(size(x));
        for mi = 1:M % models
            model = models{mi};
            fprintf('denoising image %i with %s (noise %2f).\n',...
                    i, model.name, noise)
            if isnan(ll(mi))
                ll(mi) = model.loglikelihood(tst_ps) ./...
                         (log(2) * numel(tst_ps)); %in bits/pixel
            end
            tic
            xhat = denoise(y, model, noise, psize);
            dur(i,mi,si) = toc;
            psnr(i,mi,si) = pSNR(x,xhat);
            if ~isempty(to_plot)
                lims = round((size(x) - to_plot)/2);
                lime = round((size(x) + to_plot)/2);
                axes(ahs(i,mi,si));
                imagesc(xhat(lims(1):lime(1),lims(2):lime(2)));
                set(gca, 'xtick',[],'ytick',[]);
                if mi == 1
                    title(sprintf('noise = %.2f', noise), 'fontsize', 18);
                end
                if si == 1
                    ylabel(model.name, 'fontsize', 14)
                end
            end
            pause(.1);
        end
    end
end
