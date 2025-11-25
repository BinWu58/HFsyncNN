function [delta_X_NN, Index_NN, internals] = Fun_NN_Inputation( ...
    Ylog_Nonsyn, r_dim, mu, lambda, eta2, varargin)
%FUN_NN_INPUTATION  Nuclear-norm based synchronization core routine.
%
%   [delta_X_NN, Index_NN, internals] = HFsyncNN.Fun_NN_Inputation( ...
%       Ylog_Nonsyn, r_dim, mu, lambda, eta2, ...)
%
%   This function implements the linear-constraint construction A, b,
%   synchronous filling of missing prices within Ylog_Nonsyn, PCA-based
%   initialization (or user-provided initialization), and calls the
%   ADMM solver.
%
%   INPUT
%   -----
%   Ylog_Nonsyn : T x N matrix, zeros indicate missing values.
%   r_dim       : number of common factors.
%   mu          : ADMM parameter.
%   lambda      : nuclear norm penalty.
%   eta2        : eta used in ADMM (in your notation).
%
%   Name-Value pairs:
%   'InitMode'      : 'pca' (default) or 'user'.
%   'InitialSignal' : matrix (same size as delta_X_Syn) if InitMode='user'.
%   'MaxIter'       : max ADMM iterations (default 10000).
%   'Tol'           : stopping tolerance (default 1e-4).
%   'Verbose'       : logical, print basic info (default false).
%
%   OUTPUT
%   ------
%   delta_X_NN : synchronized log-return matrix.
%   Index_NN   : indices of rows in original Ylog_Nonsyn kept by NN.
%   internals  : struct with fields:
%       .ADMMIters : iterations used.
%       .Tol       : tolerance used.
%       .delta_X_Syn, .Z_pi, .A, .b (optional for diagnostic use).

% -------------------------
% 1. parse options
% -------------------------
p = inputParser;
p.FunctionName = 'HFsyncNN.Fun_NN_Inputation';

addParameter(p, 'InitMode', 'pca', @(s) ischar(s) || isstring(s));
addParameter(p, 'InitialSignal', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'MaxIter', 10000, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Tol', 1e-4, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Verbose', false, @(x) islogical(x) || isnumeric(x));

parse(p, varargin{:});
opt = p.Results;
opt.InitMode = char(opt.InitMode);
opt.Verbose = logical(opt.Verbose);

% -------------------------
% 2. Handling rows where all values are 0
% -------------------------
if any(all(Ylog_Nonsyn == 0, 2))
    first_nonzero_idx = find(Ylog_Nonsyn(:, 1) ~= 0, 1, 'first');
    if isempty(first_nonzero_idx)
        error('HFsyncNN:Fun_NN_Inputation:InvalidInput', ...
            'First column is all zero; cannot find non-zero value to fill.');
    end

    last_nonzero_col1_val = Ylog_Nonsyn(first_nonzero_idx, 1);

    for i = 1:size(Ylog_Nonsyn, 1)
        if all(Ylog_Nonsyn(i, :) == 0)
            % small perturbation to avoid exact duplicates
            Ylog_Nonsyn(i, 1) = last_nonzero_col1_val + rand(1,1)*1e-4;
        else
            if Ylog_Nonsyn(i, 1) ~= 0
                last_nonzero_col1_val = Ylog_Nonsyn(i, 1);
            end
        end
    end
end

% -------------------------
% 3. 3. Construct A, b 
% -------------------------
X = Ylog_Nonsyn;
X(X == 0) = NaN;
[pRows, p] = size(Ylog_Nonsyn); %#ok<ASGLU>

time_all_NaN = find(sum(isnan(X), 2) == p);
X(time_all_NaN, :) = [];

Index_NN = 1:size(Ylog_Nonsyn,1);
Index_NN(ismember(Index_NN, time_all_NaN)) = [];
time_all_NaN = [];

% Get n; n_all; X_n
X_n = (1:size(X, 1))';
X_n = repmat(X_n, 1, p);
X_n(isnan(X)) = NaN;
n1 = length(rmmissing(unique(X_n))) - 1;

n_all = zeros(1, p);
for stock = 1:p
    n_all(stock) = size(rmmissing(X(:, stock)), 1) - 1;
end
n_all_sum = sum(n_all);

A = sparse(n_all_sum, n1 * p);
n_sum = 0;
for stock = 1:p
    n_stock = n_all(stock);
    col_n = rmmissing(X_n(:, stock));

    A_stock = sparse(n_stock, n1);
    for row = 1:n_stock
        pos1 = col_n(row);
        pos2 = col_n(row+1);
        if pos2 > (pos1 + 1)
            A_stock(row, pos1:(pos2-1)) = 1;
        else
            A_stock(row, pos2-1) = 1;
        end
    end

    A((n_sum + 1):(n_sum + n_stock), (stock*n1 - n1 + 1):(stock*n1)) = A_stock;
    n_sum = n_sum + n_stock;
end
A_stock = []; col_n = []; X_n = [];

% GET b
b = [];
for stock = 1:p
    col = rmmissing(X(:, stock));
    diff_col = diff(col);
    % If jump-trimming is added in the future, it can be added here.
    b = [b; diff_col]; %#ok<AGROW>
end
diff_col = []; col = [];

% Delete all rows where all assets are missing (all 0s)
I1 = (Ylog_Nonsyn == 0);
I2 = sum(I1, 2);
Index = find(I2 == p);
Ylog_Nonsyn(Index, :) = [];

% Construct X_Syn (previous tick style) to obtain delta_X_Syn
X_Syn = [];
[size1, size2] = size(Ylog_Nonsyn);
X_Syn(1,:) = Ylog_Nonsyn(1,:);
for i = 2:size1
    Y1 = Ylog_Nonsyn(i,:);
    Y2 = X_Syn(i-1,:);

    I1 = (Y1 ~= 0);
    I2 = (Y1 == 0);
    X_Syn(i,:) = Y1.*I1 + Y2.*I2;
end
delta_X_Syn = diff(X_Syn);

Z_x = delta_X_Syn;

% -------------------------
% 4. Initial value: PCA or user-specified
% -------------------------
if strcmpi(opt.InitMode, 'pca')
    [Chat, ~, ~] = HFsyncNN.Fun_PCA_C(delta_X_Syn, r_dim);
    Z_pi = Chat';
elseif strcmpi(opt.InitMode, 'user')
    if isempty(opt.InitialSignal)
        error('HFsyncNN:Fun_NN_Inputation:MissingInitialSignal', ...
            'InitMode is ''user'' but InitialSignal is empty.');
    end
    if ~isequal(size(opt.InitialSignal), size(delta_X_Syn))
        error('HFsyncNN:Fun_NN_Inputation:InitialSignalSize', ...
            'InitialSignal must have the same size as delta_X_Syn.');
    end
    Z_pi = opt.InitialSignal;
else
    error('HFsyncNN:Fun_NN_Inputation:UnknownInitMode', ...
        'InitMode must be ''pca'' or ''user''.');
end

% -------------------------
% 5.Call ADMM to solve
% -------------------------
[delta_X_NN, Pi_hat3, ite_num3] = HFsyncNN.Fun_ADMM_Modified_syn3( ...
    mu, lambda, eta2, b, A, Z_x, Z_pi, ...
    'MaxIter', opt.MaxIter, ...
    'Tol', opt.Tol, ...
    'Verbose', opt.Verbose);

% -------------------------
% 6. Output internal information
% -------------------------
internals = struct();
internals.ADMMIters   = ite_num3;
internals.Tol         = opt.Tol;
internals.delta_X_Syn = delta_X_Syn;
internals.Z_pi        = Z_pi;
internals.A           = A;
internals.b           = b;
internals.Pi_hat      = Pi_hat3;

end
