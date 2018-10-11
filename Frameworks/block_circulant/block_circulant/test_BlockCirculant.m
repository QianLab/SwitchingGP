%test_BlockCirculant
%m=methods('BlockCirculant')
type='square';
switch type
    case 'square'
        M=2;
        N=4;
        sym=2;
end
% a=rand(M,N)+sqrt(-1)*rand(M,N); %use complex matrices
a=rand(M,N);
A=BlockCirculant(a,sym);
b=rand(M,N)+sqrt(-1)*rand(M,N); %use complex matrices
B=BlockCirculant(b,sym);
a1=double(A); %recover complete matrix from circulant slice
if all(all(a1(1:M,:)==a)) %check match
    disp('match convert to double');
end

%trivial operations (plus, minus, uplus, uminus
A1p=+A;
A1m=-A;
A2=A+1;
A2=1+A;
A2=A+A;
A2=A-1;
A2=1-A;
A2=A-A;
[a2,sym2]=unpack(A); %retrieve original slice
if all(all(a2==a)) %check match
    disp('match unpack');
end
Ah=A'; %check Hermitian transpose
if all(all(double(A)'==double(Ah))) %check match
    disp('match  Hermitian transpose');
end
At=A.'; %check .' transpose
if all(all(double(A).'==double(At))) %check match
    disp('match transpose');
end

disp('multiply errors');
%BlockCirculant times ordinary matrix
b=rand(N,5)+sqrt(-1)*rand(N,5); %use complex matrices
ProdAb=A*b;
ProdAb1=double(A)*b;
disp(max(max(abs(ProdAb-ProdAb1))))

%ordinary times block circulant
ProdAbT=b.'*A.';
disp(max(max(abs(ProdAbT-ProdAb1.'))))

%block circ times block circ
ProdAB=A*B;
disp(max(max(abs(double(ProdAB)-double(A)*double(B)))))

%square matrix inverse - errors should be on the order of machine precision
AI=inv(A);
ident=double(AI*A);
disp('square matrix inverse errors')
disp(max(max(abs(ident-eye(length(double(A)))))))
ident=double(A*AI);
disp(max(max(abs(ident-eye(length(double(A)))))))

%left matrix divide
disp('left-divide errors')
Adivb=A\b;
disp(max(max(abs(Adivb-double(A)\b))))
AdivB=A\B;
disp(max(max(abs(double(AdivB)-double(A)\double(B)))))

%concatenate two block-circulant matrices
Chor=[A B];
Cvert=[A;B];

%extract submatrices by using abbreviated subscripting
Ah=Chor(1:M,1:N/sym);
if all(all(double(A)==double(Ah))) %check match
    disp('match subscript 1 of 4');
end
Bh=Chor(1:M,N/sym+(1:N/sym));
if all(all(double(B)==double(Bh))) %check match
    disp('match subscript 2 of 4');
end
Av=Cvert(1:M,1:N/sym);
if all(all(double(A)==double(Av))) %check match
    disp('match subscript 3 of 4');
end
Bv=Cvert(M+(1:M),1:N/sym);
if all(all(double(B)==double(Bv))) %check match
    disp('match subscript 4 of 4');
end

%check pseudoinverse
Ap=pinv(Cvert);
disp('pseudoinverse errors')
disp('long thin')
disp(max(max(abs(double(Ap)-pinv(double(Cvert))))))
Ap=pinv(Chor);
disp('short fat')
disp(max(max(abs(double(Ap)-pinv(double(Chor))))))

%check submatrix calculations
As=submatrix(A,1:M*sym,1:N);
if all(all(As==double(A))) %check match
    disp('match submatrix');
end

%check special case pseudoinverse
r=rand(1,12);
M=BlockCirculant(r,12);
M1=pinv(M);
M1exp=pinv(double(M));
disp('error for pure curculant (rxr matrix represented by 1xr, symmetry r:')
disp(max(max(abs(M1exp-double(M1)))))

