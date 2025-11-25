function [delta_X_hat, Pi_hat, k] = Fun_ADMM_Modified_syn3( ...
    mu, lambda, eta, b, A, Z_x, Z_pi, varargin)
%FUN_ADMM_MODIFIED_SYN3  ADMM solver for nuclear-norm minimization.
%
%   [delta_X_hat, Pi_hat, k] = HFsyncNN.Fun_ADMM_Modified_syn3( ...
%       mu, lambda, eta, b, A, Z_x, Z_pi, ...)
%
%   Name-Value pairs:
%   'MaxIter' : maximum number of iterations (default 10000).
%   'Tol'     : convergence tolerance (default 1e-4).
%   'Verbose' : logical, whether to print progress (default false).

% parse optional parameters
p = inputParser;
p.FunctionName = 'HFsyncNN.Fun_ADMM_Modified_syn3';
addParameter(p, 'MaxIter', 10000, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Tol', 1e-4, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Verbose', false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
opt = p.Results;
MAX_ITER = opt.MaxIter;
Tol = opt.Tol;
verbose = logical(opt.Verbose);

[n, pDim] = size(Z_x);

% Precompute matrices (Woodbury-like trick)
M0_diag = 1 ./ (1 + (2/eta) * sum(A.^2, 2));
M0 = spdiags(M0_diag, 0, size(A, 1), size(A, 1));

M = (2/eta) * speye(n*pDim);
M = M - (2/eta)^2 * (A' * M0 * A);
M = M / 2;

M1 = 2 * A' * b;

% Initialization
U_x  = ones(n, pDim);   % or zeros(n,pDim), your original code used ones
U_pi = zeros(n, pDim);

% Iterations
for k = 1:MAX_ITER
    if k == 1
        Vec_delta_X = M * (M1 + eta * Z_x(:) + eta * U_x(:));
        delta_X = reshape(Vec_delta_X, n, pDim);
        Pi = (2*mu*Z_x + eta*Z_pi + eta*U_pi) / (2*mu + eta);
    else
        delta_X = delta_X_new;
        Pi      = Pi_new;
    end

    Z_x  = (2*mu*Pi + eta*delta_X - eta*U_x) / (2*mu + eta);
    Z_pi = shrink(Pi - U_pi, lambda / eta);

    U_x  = U_x  + Z_x  - delta_X;
    U_pi = U_pi + Z_pi - Pi;

    % Next step, delta_X and Pi
    Vec_delta_X_new = M * (M1 + eta * Z_x(:) + eta * U_x(:));
    delta_X_new     = reshape(Vec_delta_X_new, n, pDim);
    Pi_new          = (2*mu*Z_x + eta*Z_pi + eta*U_pi) / (2*mu + eta);

    % Normalized errors
    error1 = norm(delta_X_new - delta_X, 'fro') / ...
        max([norm(delta_X_new, 'fro'), norm(delta_X, 'fro'), 1]);
    error2 = norm(Pi_new - Pi, 'fro') / ...
        max([norm(Pi_new, 'fro'),norm(Pi, 'fro'), 1]);
    error3 = norm(Z_x - delta_X, 'fro') / ...
        max([norm(Z_x, 'fro'), norm(delta_X, 'fro'), 1]);
    error4 = norm(Z_pi - Pi, 'fro') / ...
        max([norm(Z_pi, 'fro'), norm(Pi, 'fro'), 1]);

    if max([error1,error2,error3,error4]) < Tol
        if verbose
            fprintf('ADMM converged at iteration %d, max error = %.2e\n', ...
                k, max([error1,error2,error3,error4]));
        end
        delta_X_hat = delta_X_new;
        Pi_hat      = Pi_new;
        return;
    end

    delta_X_hat = delta_X_new;
    Pi_hat      = Pi_new;

end

if verbose
    fprintf('ADMM reached MAX_ITER=%d, final max error = %.2e\n', ...
        MAX_ITER, max([error1,error2,error3,error4]));
end

end

% -------------------------------
% Gandy (2011) equation (8): singular-value soft-thresholding
% -------------------------------
function shrink_result = shrink(T, tau)
[U, Sigma, V] = svd(T, 'econ');
Sigma_tilde = max(diag(Sigma) - tau, 0);
Sigma_tilde = diag(Sigma_tilde);
shrink_result = U * Sigma_tilde * V';
end
