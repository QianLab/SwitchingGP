function yn=iscirculant(A,N,tol)
%function yn=iscirculant(A,k,tol) - test a full size matrix (kM by kN) to see if is has a circulant form.
%
%A is the full size matrix, k is the symmetry order, tol is an optional tolerance (if all absolute values of differences
%   are less than tol, the matrix passes the test).
if ~exist('tol','var')
    tol=1E-12;
end
D=size(A);
if any(D/N~=floor(D/N))
    yn=0;
    return
end
D1=D/N;
A1=A(1:D1(1),:);
mA=max(max(abs(A1)));
for i=2:N
    A2=A((i-1)*D1(1)+[1:D1(1)],:);
    err=max(max(abs(rotate_matrix(A2,D1(2)*(i-1),2)-A1)))/mA;
    if err>tol
        yn=0;
        return
    end
end
yn=1;
end
