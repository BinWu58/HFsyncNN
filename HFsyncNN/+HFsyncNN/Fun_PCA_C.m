function [C_hat, Xb_hat, beta_hat] = Fun_PCA_C(Yb, r_dim)
%FUN_PCA_C  PCA-based low-rank approximation of returns.
%
%   [C_hat, Xb_hat, beta_hat] = HFsyncNN.Fun_PCA_C(Yb, r_dim)
%
%   INPUT
%   -----
%   Yb      : n x p matrix of returns (in your NN code this is delta_X_Syn).
%   r_dim   : number of factors.
%
%   OUTPUT
%   ------
%   C_hat   : p x n low-rank reconstruction (beta_hat * Xb_hat).
%   Xb_hat  : r x n factor return matrix.
%   beta_hat: p x r loading matrix.

Yb = Yb';  % now p x n

[p, kn] = size(Yb);
C_hat = zeros(p, kn); %#ok<NASGU>

% Symmetrized covariance
M = Yb * Yb' / (p * kn);
M = (M + M') / 2;

% Leading r_dim eigenvectors
[beta_hat, ~] = eigs(M, r_dim);

% Factor scores
Xb_hat = beta_hat' * Yb;    % r x n
C_hat  = beta_hat * Xb_hat; % p x n

end
