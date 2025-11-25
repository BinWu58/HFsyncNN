function [delta_X_NN, X_Syn_NN, info] = nnSync(Ylog_Nonsyn, varargin)
%NNSYNC  Nuclear-norm based synchronization of asynchronous log-price data.
%
%   [delta_X_NN, X_Syn_NN, info] = HFsyncNN.nnSync(Ylog_Nonsyn, ...)
%
%   This function implements the nuclear-norm-based synchronization
%   method for asynchronous high-frequency log-prices, as developed in
%   "Data Synchronization at High Frequencies".
%
%   INPUT
%   -----
%   Ylog_Nonsyn : T x N double
%       Asynchronous log-price matrix. Missing observations must be
%       encoded as 0. The first row in each column must be non-zero.
%
%   Name-Value pair arguments:
%
%   'NumFactors'     : integer, number of common factors (r_dim).
%                      Default: 3.
%
%   'Mu'             : double, ADMM penalty parameter for coupling term µ.
%                      Default: 0.1.
%
%   'Lambda'         : double, nuclear-norm penalty λ.
%                      Default: 0.001.
%
%   'Tau'            : double, scaling parameter used in eta2 = lambda / tau.
%                      Default: 0.1.
%
%   'UseReturns'     : logical, whether to reconstruct synchronized
%                      log-prices using the NN returns via
%                      fillLogPriceFromReturn.
%                      true  -> X_Syn_NN is returned;
%                      false -> X_Syn_NN = [].
%                      Default: true.
%
%   'InitMode'       : 'pca' or 'user'.
%                      'pca'  -> use PCA-based initialization (Fun_PCA_C).
%                      'user' -> use user-provided low-rank initial signal.
%                      Default: 'pca'.
%
%   'InitialSignal'  : n_syn x N matrix (only used if InitMode='user').
%                      This should be an initial guess for the
%                      low-rank component in the synchronized returns
%                      (same size as delta_X_Syn inside the algorithm).
%
%   'MaxIter'        : integer, maximum number of ADMM iterations.
%                      Default: 10000.
%
%   'Tol'            : double, convergence tolerance for ADMM updates.
%                      Default: 1e-4.
%
%   'Verbose'        : logical, whether to print basic progress information.
%                      Default: false.
%
%   OUTPUT
%   ------
%   delta_X_NN : n_syn x N double
%       Synchronized log-return matrix reconstructed by nuclear-norm
%       minimization under linear constraints A * vec(Δ)' = b.
%
%   X_Syn_NN   : (n_syn+1) x N double
%       Synchronized log-price matrix obtained by filling missing prices
%       using delta_X_NN (only if UseReturns = true). If UseReturns = false,
%       X_Syn_NN is returned as [].
%
%   info       : struct with additional output:
%       .Index_NN      : indices of rows in Ylog_Nonsyn used by NN method.
%       .ADMMIters     : number of ADMM iterations used.
%       .ADMMTol       : tolerance for stopping.
%       .InitMode      : 'pca' or 'user'.
%       .NumFactors    : r_dim.
%       .Mu, .Lambda, .Tau, .Eta2 : tuning parameters actually used.
%
%   Example:
%   --------
%   load example_Ylog_Nonsyn.mat  % T x N matrix with zeros as missing
%   [dX_NN, X_NN, info] = HFsyncNN.nnSync(Ylog_Nonsyn, ...
%       'NumFactors', 3, 'Mu', 0.1, 'Lambda', 0.001, 'Tau', 0.1, ...
%       'UseReturns', true);
%
%   See also: HFsyncNN.Fun_NN_Inputation, HFsyncNN.fillLogPriceFromReturn

% -------------------------
% 1. default options
% -------------------------
p = inputParser;
p.FunctionName = 'HFsyncNN.nnSync';

addRequired(p, 'Ylog_Nonsyn', @(x) isnumeric(x) && ismatrix(x));

addParameter(p, 'NumFactors', 3, @(x) isnumeric(x) && isscalar(x) && x>=1);
addParameter(p, 'Mu', 0.1, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Lambda', 0.001, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Tau', 0.1, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'UseReturns', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'InitMode', 'pca', @(s) ischar(s) || isstring(s));
addParameter(p, 'InitialSignal', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'MaxIter', 10000, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Tol', 1e-4, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Verbose', false, @(x) islogical(x) || isnumeric(x));

parse(p, Ylog_Nonsyn, varargin{:});
opt = p.Results;

opt.InitMode = char(opt.InitMode);
opt.UseReturns = logical(opt.UseReturns);
opt.Verbose = logical(opt.Verbose);

% eta2 in your original code
eta2 = opt.Lambda / opt.Tau;

% -------------------------
% 2. call NN core
% -------------------------
[delta_X_NN, Index_NN, internals] = HFsyncNN.Fun_NN_Inputation( ...
    Ylog_Nonsyn, opt.NumFactors, opt.Mu, opt.Lambda, eta2, ...
    'InitMode', opt.InitMode, ...
    'InitialSignal', opt.InitialSignal, ...
    'MaxIter', opt.MaxIter, ...
    'Tol', opt.Tol, ...
    'Verbose', opt.Verbose);

% -------------------------
% 3. optionally reconstruct prices
% -------------------------
if opt.UseReturns
    X_Syn_NN = HFsyncNN.fillLogPriceFromReturn(Ylog_Nonsyn, delta_X_NN);
else
    X_Syn_NN = [];
end

% -------------------------
% 4. collect info
% -------------------------
info = struct();
info.Index_NN   = Index_NN;
info.ADMMIters  = internals.ADMMIters;
info.ADMMTol    = internals.Tol;
info.InitMode   = opt.InitMode;
info.NumFactors = opt.NumFactors;
info.Mu         = opt.Mu;
info.Lambda     = opt.Lambda;
info.Tau        = opt.Tau;
info.Eta2       = eta2;

end
