% HFsyncNN Toolbox
% Version 1.0  2025-xx-xx
%
% Nuclear-norm based synchronization of asynchronous high-frequency
% log-price data via large-scale linear systems and ADMM.
%
% Main user-facing function:
%   HFsyncNN.nnSync           - High-level interface for NN synchronization.
%
% Core internal functions:
%   HFsyncNN.Fun_NN_Inputation   - Core NN synchronization routine.
%   HFsyncNN.Fun_ADMM_Modified_syn3 - ADMM solver for nuclear-norm problem.
%   HFsyncNN.Fun_PCA_C           - PCA-based initialization for low-rank component.
%   HFsyncNN.fillLogPriceFromReturn - Optional reconstruction of log-prices
%                                     from synchronized returns.
%
% Example scripts:
%   examples/demo_basic.m        - Minimal example with simulated data.
%
% See README.md for detailed usage, installation, and data format.
