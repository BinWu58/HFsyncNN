% DEMO_BASIC  Minimal example for HFsyncNN.nnSync
%
% This script simulates simple underlying log-prices, generates
% asynchronous observations with zeros indicating missing values, and
% then applies the nuclear-norm synchronization method.
%
% Asynchronicity is controlled by two parameters (L, U):
%   - L: lower bound of asynchronicity (lowest missing intensity)
%   - U: upper bound of asynchronicity (highest missing intensity)
% For asset j = 1,...,N, the drop intensity (expected missing ratio)
% is defined as a linear grid from U (asset 1) down to L (asset N).
%
% The construction of Ylog_Nonsyn follows the principles of the
% original simulation code:
%   - log-prices start from a non-zero initial value;
%   - asynchronous sampling may generate rows where all assets are 0;
%   - for each asset, the first row is forced to be non-zero by
%     copying the first observed non-zero log-price in that column.

clear; clc;

% -------------------------
% 0. user-chosen settings
% -------------------------
T  = 200;   % number of return steps
N  = 100;     % number of assets

% Asynchronicity intensity bounds:
% Larger value => more missing observations (more asynchronous).
L = 0.10;   % lower bound of drop intensity (asset N)
U = 0.60;   % upper bound of drop intensity (asset 1)

% Asset indices to be plotted (highest and lowest asynchronicity)
assetHighA = 1;   % asset with highest asynchronicity (drop intensity U)
assetLowA  = N;   % asset with lowest asynchronicity (drop intensity L)

rng(1);     % for reproducibility

% -------------------------
% 1. simulate underlying log-prices
% -------------------------
% simple random-walk dynamics with different vol levels
sigma = linspace(0.01, 0.03, N);   % asset-specific volatilities
dW    = randn(T, N) .* sigma;      % increments

logP0 = log(100);                  % non-zero initial log-price
logP  = zeros(T+1, N);
logP(1, :) = logP0;

for t = 2:T+1
    logP(t, :) = logP(t-1, :) + dW(t-1, :);
end
% logP is (T+1) x N, all entries non-zero with probability 1

% -------------------------
% 2. generate asynchronous sampling: zeros = missing values
% -------------------------
Ylog_Nonsyn = logP;                % start from fully observed panel
[T1, Ncheck] = size(logP);         % T1 = T+1
if Ncheck ~= N
    error('Dimension mismatch between T/N and logP.');
end

% Define drop intensities (asynchronicity) per asset:
% asset 1 has intensity U (most missing), asset N has L (least missing)
dropIntensity = linspace(U, L, N);   % 1 x N
obsProb       = 1 - dropIntensity;   % observation probabilities

for j = 1:N
    % random mask: true = drop (set to zero), false = keep
    dropMask = rand(T1, 1) < dropIntensity(j);  % expected missing ratio dropIntensity(j)
    Ylog_Nonsyn(dropMask, j) = 0;

    % ensure the first row is non-zero for this asset:
    % if time 1 was dropped, copy the first non-zero observation
    % upward to time 1 (as in the original simulation code).
    if Ylog_Nonsyn(1, j) == 0
        idx = find(Ylog_Nonsyn(:, j) ~= 0, 1, 'first');
        if ~isempty(idx)
            Ylog_Nonsyn(1, j) = Ylog_Nonsyn(idx, j);
        else
            % safety fallback: if the entire column is zero (extremely
            % unlikely in practice), keep the original fully observed
            % log-price path for this asset.
            Ylog_Nonsyn(:, j) = logP(:, j);
        end
    end
end

% -------------------------
% 2.1 print basic data and asynchronicity information
% -------------------------
totalEntries      = T1 * N;
numMissing        = nnz(Ylog_Nonsyn == 0);
missingRatio      = numMissing / totalEntries;
numAllZeroRows    = sum(all(Ylog_Nonsyn == 0, 2));
missingPerAsset   = sum(Ylog_Nonsyn == 0, 1);
missingRatioAsset = missingPerAsset / T1;

fprintf('================ DATA SUMMARY ================\n');
fprintf('Number of time points (T+1)      : %d\n', T1);
fprintf('Number of assets (N)             : %d\n', N);
fprintf('Total number of entries          : %d\n', totalEntries);
fprintf('Number of missing entries        : %d\n', numMissing);
fprintf('Missing ratio (overall)          : %.4f\n', missingRatio);
fprintf('Number of all-zero rows          : %d\n', numAllZeroRows);
fprintf('----------------------------------------------\n');
fprintf('Asynchronicity bounds (drop intensity):\n');
fprintf('  Lower bound L (asset %d)      : %.4f\n', assetLowA,  L);
fprintf('  Upper bound U (asset %d)      : %.4f\n', assetHighA, U);
fprintf('Drop intensity and missing ratio per asset:\n');
for j = 1:N
    fprintf('  Asset %d: dropIntensity = %.4f, missing = %4d (ratio = %.4f)\n', ...
        j, dropIntensity(j), missingPerAsset(j), missingRatioAsset(j));
end
fprintf('==============================================\n\n');

% -------------------------
% 3. call NN synchronization (measure runtime)
% -------------------------
fprintf('Running HFsyncNN.nnSync ...\n');
tic;
[delta_X_NN, X_Syn_NN, info] = HFsyncNN.nnSync(Ylog_Nonsyn, ...
    'NumFactors', 2, ...
    'Mu', 0.1, ...
    'Lambda', 0.001, ...
    'Tau', 0.1, ...
    'UseReturns', true, ...
    'Verbose', true);
elapsedTime = toc;

fprintf('NN synchronization completed.\n');
fprintf('  ADMM iterations    : %d\n', info.ADMMIters);
fprintf('  Runtime (seconds)  : %.4f\n\n', elapsedTime);

% -------------------------
% 4. visualization:
%    For the asset with highest asynchronicity (assetHighA)
%    and the asset with lowest asynchronicity (assetLowA),
%    plot:
%      (a) true vs asynchronous log-price
%      (b) true synchronized vs NN synchronized log-price
% -------------------------
figure;

% ----- Asset with highest asynchronicity (assetHighA) -----
subplot(2,2,1);
plot(logP(:,assetHighA), 'LineWidth', 1.0); hold on;
stem(find(Ylog_Nonsyn(:,assetHighA)~=0), ...
     Ylog_Nonsyn(Ylog_Nonsyn(:,assetHighA)~=0,assetHighA), '.');
title(sprintf('Asset %d (highest async): True vs Asynch', assetHighA));
legend('True log-price','Observed (asynchronous)', ...
       'Location','Best');
xlabel('Time index');
ylabel('Log-price');

subplot(2,2,3);
plot(logP(:,assetHighA), '--', 'LineWidth', 1.0); hold on;
plot(X_Syn_NN(:,assetHighA), 'LineWidth', 1.2);
title(sprintf('Asset %d: True vs NN synchronized', assetHighA));
legend('True synchronized','NN synchronized', ...
       'Location','Best');
xlabel('Time index');
ylabel('Log-price');

% ----- Asset with lowest asynchronicity (assetLowA) -----
subplot(2,2,2);
plot(logP(:,assetLowA), 'LineWidth', 1.0); hold on;
stem(find(Ylog_Nonsyn(:,assetLowA)~=0), ...
     Ylog_Nonsyn(Ylog_Nonsyn(:,assetLowA)~=0,assetLowA), '.');
title(sprintf('Asset %d (lowest async): True vs Asynch', assetLowA));
legend('True log-price','Observed (asynchronous)', ...
       'Location','Best');
xlabel('Time index');
ylabel('Log-price');

subplot(2,2,4);
plot(logP(:,assetLowA), '--', 'LineWidth', 1.0); hold on;
plot(X_Syn_NN(:,assetLowA), 'LineWidth', 1.2);
title(sprintf('Asset %d: True vs NN synchronized', assetLowA));
legend('True synchronized','NN synchronized', ...
       'Location','Best');
xlabel('Time index');
ylabel('Log-price');
