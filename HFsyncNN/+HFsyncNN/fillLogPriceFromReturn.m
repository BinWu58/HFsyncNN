function Y_filled = fillLogPriceFromReturn(Y, X)
%FILLLOGPRICEFROMRETURN  Fill missing log-prices using log-returns.
%
%   Y_filled = HFsyncNN.fillLogPriceFromReturn(Y, X)
%
%   INPUT
%   -----
%   Y : (n+1) x N matrix of log-prices, where 0 indicates missing values
%       (except the first row, which must be non-zero for each asset).
%   X : n x N matrix of synchronized log-returns.
%
%   OUTPUT
%   ------
%   Y_filled : (n+1) x N matrix with missing entries filled based on
%              cumulative log-returns.

[n1, N] = size(Y);
n = n1 - 1;
Y_filled = Y;

for j = 1:N
    i = 2;
    while i <= n1
        if Y(i, j) == 0
            % start of a missing block
            start_idx = i - 1;
            if Y(start_idx, j) == 0
                error('fillLogPriceFromReturn:InvalidInput', ...
                      'Missing value without known prior price in column %d.', j);
            end
            % find the end of the block
            end_idx = i;
            while end_idx <= n1 && Y(end_idx, j) == 0
                end_idx = end_idx + 1;
            end
            % fill using cumulative returns
            for k = i:(end_idx - 1)
                Y_filled(k, j) = Y_filled(k - 1, j) + X(k - 1, j);
            end
            i = end_idx;
        else
            i = i + 1;
        end
    end
end

end
