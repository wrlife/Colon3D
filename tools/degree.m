% D = degree(W)
%   Compute the degree matrix from an adjacency
%   W    - a matrix
%   D    - diagonal degree matrix (each elt is the column sum of W)
%   Dinv - inverse diagonal degree matrix
%
function [D,Dinv] = degree(W)

    n    = size(W,1);
    D    = spdiags(sum(W)',    0,speye(n)); %D = diag(sum(W));
    Dinv = spdiags(sum(W)'.^-1,0,speye(n)); %D = diag(sum(W));
    
end
