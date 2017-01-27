load ims.mat
close all;

K = 5;
options.verbosity = 'iter';

X_train = standardize_ims(ims.train);
X_test = standardize_ims(ims.train);

patches = sample_patches(X_train);

mvn_model = learn_MVN(patches, options);
mvn.denoise = @(Y, noise) MVN_denoise(Y, mvn_model, noise);
mvn.loglikelihood = @(X) MVN_loglikelihood(X, mvn_model);
mvn.name = 'MVN';

ica_model = learn_ICA(patches, K, options);
ica.denoise = @(Y, noise) ICA_denoise(Y, ica_model, noise);
ica.loglikelihood = @(X) ICA_loglikelihood(X, ica_model);
ica.name = 'ICA';

gsm_model = learn_GSM(patches, K, options);
gsm.denoise = @(Y, noise) GSM_denoise(Y, gsm_model, noise);
gsm.loglikelihood = @(X) GSM_loglikelihood(X, gsm_model);
gsm.name = 'GSM';

models = {mvn, ica, gsm};

[psnr, ll, dur] = test_denoising(X_test, models);

fprintf('pSNR results:')
reshape(mean(psnr, 1), 4, 3)

fprintf('Log-likelihood results:')
ll

fprintf('Total denoise duration (seconds):')
sum(dur(:))
