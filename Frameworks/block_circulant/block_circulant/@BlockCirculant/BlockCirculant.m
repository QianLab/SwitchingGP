%function m=BlockCirculant(m,symmetry)
%Takes a single "row slice" of a block-circulant matrix (e.g., m=ABCD) and stores it
%and its symmetry order as a BlockCirculant object, M. The BlockCirculant
%object represents the full matrix, e.g.,
%
%   ABCD
%   DABC
%   CDAB
%   BCDA
%
%The sub-matrices (blocks) can be any shape.
%
%For example, if the individual block size is 4x3 and the symmetry order is
%4, the size of matrix m would be 4x12. This would represent a matrix ABCD
%where A, B, C, and D are 4x3. The matrix object would be the compact
%representation of a 16x12 block-circulant matrix.
%
%The usual NxN fully circulant matrix used in convolutions, etc., would have a
%block size of 1x1 and would be represented by a matrix m with dimension 1xN.
classdef BlockCirculant
    properties
        value
        symmetry
    end
    methods
        %------------------------------------------------------------
        function M=BlockCirculant(m,sym) %constructor
            if length(size(m))~=2
                error('BlockCirculant: matrix to use as block circulant slice should be dimension 2');
            end
            [~,Nb]=size(m);
            if ~exist('sym','var') || ~(0==mod(Nb,sym))
                error('BlockCirculant: number of blocks not defined or non-integer number of blocks (#columns/symmetry not an integer');
            end
            M.value=m;
            M.symmetry=sym;
            %M=class(M,'BlockCirculant');
        end
        %------------------------------------------------------------
        function [a,sym]=unpack(A)
            %function [a,sym]=unpack(A)
            %retrieve the original circulant blocks from a matrix object
            a=A.value;
            sym=A.symmetry;
        end
        %------------------------------------------------------------
        function C=plus(A,B) %C=A+B
            if isnumeric(B) && isscalar(B)
                C=A;
                C.value=C.value+B;
                return
            elseif isnumeric(A) && isscalar(A)
                C=B;
                C.value=C.value+A;
                return
            elseif ~isa(A,'BlockCirculant') || ~isa(B,'BlockCirculant')
                error('BlockCirculant: two operands do not conform or are not BlockCirculant or scalar')
            end
            if A.symmetry~=B.symmetry
                error('BlockCirculant: two operands do not conform (symmetry orders are different')
            end
            try
                C=A;
                C.value=C.value+B.value;
            catch
                error('BlockCirculant: two operands do not conform')
            end
        end
        %------------------------------------------------------------
        function A=uminus(B) %A=-B
            A=B;
            A.value=-A.value;
        end
        %------------------------------------------------------------
        function A=uplus(B) %A=+B
            A=B;
        end
        %------------------------------------------------------------
        function C=minus(A,B) %C=A-B
            if isnumeric(B) && isscalar(B)
                C=A;
                C.value=C.value-B;
                return
            elseif isnumeric(A) && isscalar(A)
                C=B;
                C.value=A-C.value;
                return
            elseif ~isa(A,'BlockCirculant') || ~isa(B,'BlockCirculant')
                error('BlockCirculant: two operands do not conform or are not BlockCirculant or scalar')
            end
            if A.symmetry~=B.symmetry
                error('BlockCirculant: two operands do not conform (symmetry orders are different')
            end
            try
                C=A;
                C.value=C.value-B.value;
            catch
                error('BlockCirculant: two operands do not conform')
            end
        end
        %------------------------------------------------------------
        function C=mtimes(A,B)
            %case 1: BlockCirculant*scalar or ordinary matrix
            if isnumeric(B)
                if isscalar(B)
                    C=A;
                    C.value=C.value*B; %scalar multiply
                    return
                else %circulant matrix times regular matrix/vector
                    if size(B,1)~=size(A.value,2) %nonconformal
                        error('BlockCirculant: matxices do not conform');
                    end
                    C=circulant_multiply(A.value,B,A.symmetry); %result is ordinary
                    return
                end
            end
            
            %case 2: ordinary times block circulant
            %if B is not numeric it may be block circulant
            if isnumeric(A) %then B must be block circulant
                C=(B'*A')'; %reverse order of matrix multiply
                return
            end
            
            %both block circulant?
            if ~isa(A,'BlockCirculant') || ~isa(B,'BlockCirculant')
                error('BlockCirculant: invalid object to multiply by block circulant matrix')
            end
            
            %yes, case 3: both circulant
            %make the "horizontal  slice" vertical
            if size(B.value,1)*B.symmetry~=size(A.value,2)
                error('BlockCirculant: matrices not conformal in multiply');
            end
            M=size(B.value,1);
            sym=B.symmetry;
            N=size(B.value,2)/sym;
            b=reshape(B.value,[M N sym]); %stack the blocks like a deck of cards
            b=b(:,:,[1 end:-1:2]); %reorder where they belong as if in column 1
            b=permute(b,[1 3 2]); %transpose each card
            b=reshape(b,[M*sym N]); %place turned "cards" end-to-end and flip to be a column matrix
            c=A*b; %now treat it as a BlockCirculant times an ordinary matrix (case 1 recurse)
            Nc=size(c);
            c=reshape(c,[Nc(1)/sym sym Nc(2)]);
            c=permute(c,[1 3 2]);
            c=c(:,:,[1 end:-1:2]); %reorder where they belong as if in column 1
            Nc=size(c);
            c=reshape(c,Nc(1),Nc(2)*Nc(3));
            C=BlockCirculant(c,sym);
        end
        %------------------------------------------------------------
        function C=mldivide(A,B) %C=A\B
            if ~isa(A,'BlockCirculant')
                error('BlockCirculant: in A\B, A must be block circulant')
            end
            
            %case 1: BlockCirculant\ordinary matrix
            if isnumeric(B)
                if size(B,1)~=size(A.value,2) %nonconformal
                    error('BlockCirculant: matxices do not conform');
                else
                    C=circulant_solve(A.value,B,A.symmetry); %result is ordinary
                    return
                end
            end
            
            %both circulant
            %make the "horizontal  slice" vertical
            if size(B.value,1)*B.symmetry~=size(A.value,2)
                error('BlockCirculant: matrices not conformal in multiply');
            end
            M=size(B.value,1);
            sym=B.symmetry;
            N=size(B.value,2)/sym;
            b=reshape(B.value,[M N sym]); %stack the blocks like a deck of cards
            b=b(:,:,[1 end:-1:2]); %reorder where they belong as if in column 1
            b=permute(b,[1 3 2]); %transpose each card
            b=reshape(b,[M*sym N]); %place turned "cards" end-to-end and flip to be a column matrix
            c=A\b; %now treat it as a BlockCirculant times an ordinary matrix (case 1 recurse)
            Nc=size(c);
            c=reshape(c,[Nc(1)/sym sym Nc(2)]);
            c=permute(c,[1 3 2]);
            c=c(:,:,[1 end:-1:2]); %reorder where they belong as if in column 1
            Nc=size(c);
            c=reshape(c,Nc(1),Nc(2)*Nc(3));
            C=BlockCirculant(c,sym);
        end
        %------------------------------------------------------------
        function A=pinv(B,tol)
            if ~exist('tol','var')
                tol=[];
            end
            A=B;
            A.value=circulant_pinv(A.value,A.symmetry,tol);
        end
        %------------------------------------------------------------
        function A=inv(B) %matrix inverse
            A=B;
            A.value=circulant_invert(A.value,A.symmetry);
				end
				%------------------------------------------------------------
        function D=log_det(B) %matrix inverse
            D=circulant_log_det(B.value,B.symmetry);
				end
				%------------------------------------------------------------
        function S=circ_sqrt(B) %matrix inverse
            S=circulant_sqrt(B.value,B.symmetry);
						S=BlockCirculant(S, B.symmetry);
				end
        %------------------------------------------------------------
        function B=transpose(A) %B=A'
            B=A;
            B.value=circulant_transpose(B.value,B.symmetry);
        end
        %------------------------------------------------------------
        function B=ctranspose(A) %B=A.'
            B=A;
            B.value=circulant_transpose(B.value,B.symmetry,1);
        end
        %------------------------------------------------------------
        function m=double(M)
            m=circulant_full(M.value,M.symmetry);
        end
        %------------------------------------------------------------
        function ix=end(A,k,n) % extract A(:,end), etc
            if k==1
                ix=size(A.value,1)*A.symmetry;
            else
                ix=size(A.value,2);
            end
        end
        %------------------------------------------------------------
        function A=horzcat(B,varargin) %A=[B C D...]
            %note: this maintains the circulant character by concatenating
            %submatrics
            if ~isa(B,'BlockCirculant')
                error('block circulant: argument 1 of a concatenation is not block-circulant')
            end
            Ncol=size(B.value,2);
            Nrow=size(B.value,1);
            sym=B.symmetry;
            S=0;
            for i=1:length(varargin)
                C=varargin{i};
                if ~isa(varargin{i},'BlockCirculant')
                    error('block circulant: argument %d of a concatenation is not block-circulant',i)
                end
                if size(C.value,1)~=Nrow
                    error('block circulant: argument %d of a concatenation is not conformal',i)
                end
                if sym~=C.symmetry
                    error('block circulant: argument %d of a concatenation hasdifferent order of symmetry',i)
                end
                S=S+size(C.value,2);
            end
            NewCol=S+Ncol;
            A=zeros(Nrow,NewCol);
            cols=1:size(B.value,2)/sym;
            ix=sum(gen_all_combinations({0:NewCol/sym:NewCol-1,cols}),2);
            A(:,ix)=B.value; %use circulant subscripting to add 1st matrix
            start=Ncol/sym;
            for i=1:length(varargin)
                C=varargin{i};
                cols=start+(1:size(C.value,2)/sym);
                ix=sum(gen_all_combinations({0:NewCol/sym:NewCol-1,cols}),2);
                A(:,ix)=C.value; %add additional matrices using circlant subscripting
                start=start+size(C.value,2)/sym;
            end
            A=BlockCirculant(A,sym);
        end
        %------------------------------------------------------------
        function A=vertcat(B,varargin) %A=[B;C;D...] %matrices are interleaved to preserve circulancy
            if ~isa(B,'BlockCirculant')
                error('block circulant: argument 1 of a concatenation is not block-circulant')
            end
            Ncol=size(B.value,2);
            Nrow=size(B.value,1);
            sym=B.symmetry;
            S=0;
            for i=1:length(varargin)
                C=varargin{i};
                if ~isa(varargin{i},'BlockCirculant')
                    error('block circulant: argument %d of a concatenation is not block-circulant',i)
                end
                if size(C.value,2)~=Ncol
                    error('block circulant: argument %d of a concatenation is not conformal',i)
                end
                if sym~=C.symmetry
                    error('block circulant: argument %d of a concatenation hasdifferent order of symmetry',i)
                end
                S=S+size(C.value,1);
            end
            A=B;
            v=A.value;
            v(end+S,end)=0; %extend matrix for efficient growth
            S=Nrow;
            for i=1:length(varargin)
                C=varargin{i};
                v(S+(1:size(C.value,1)),:)=C.value;
                S=S+size(C.value,1);
            end
            A.value=v;
				end
        
        %------------------------------------------------------------
        function A=submatrix(B,rows,cols) %A=B(rows,cols)
            Ncol=size(B.value,2);
            Nrow=size(B.value,1);
            sym=B.symmetry;
            if any(rows~=floor(rows)) || any(rows<1) || any(rows>Nrow*sym)
                error('block circulant: row subscript is not integer or not in range of full matrix size')
            end
            if any(cols~=floor(cols)) || any(cols<1) || any(cols>Ncol)
                error('block circulant: column subscript is not integer or not in range of full matrix size')
            end
            ix=gen_all_combinations({int32(rows(:))',int32(cols(:))'},1)-1; %form all pairs for 0-based
            %develop block in the row dimension
            voff=(ix(:,1)-mod(ix(:,1),int32(Nrow)))/int32(Nrow); %funny code is the only way to get, say, 3/2=1 in integer
            %guarantee that row number is in the first block
            ix(:,1)=mod(ix(:,1),int32(Nrow));
            %offset column to shift block in slice
            ix(:,2)=mod(ix(:,2)-voff*int32(Ncol/sym),Ncol);
            ix=ix+1;
            ix=sub2ind([Nrow,Ncol],ix(:,1),ix(:,2));
            A=flatten(B.value);
            A=reshape(A(ix),length(rows),length(cols));
        end
        %------------------------------------------------------------
        function A = subsasgn(A,S,B) %A(S)=B only; no block circulant cell arrays
            if ~isa(B,'BlockCirculant')
                error('block circulant: source in subscripted assign is not block-circulant')
            end
            if ~isa(A,'BlockCirculant')
                error('block circulant: destination in subscripted assign is not block-circulant')
            end
            Ncol=size(A.value,2);
            Nrow=size(A.value,1);
            sym=A.symmetry;
            if sym~=B.symmetry
                error('block circulant: source and destination in assign have different order of symmetry')
            end
            switch S.type
                case '()'
                    if length(S.subs{1})==1 && ischar(S.subs{1}) && S.subs{1}==':'
                        rows=1:Nrow;
                    else
                        rows=S.subs{1};
                    end
                    if length(S.subs{2})==1 && ischar(S.subs{2}) &&S.subs{2}==':'
                        cols=1:Ncol/sym; %only columns in block, not total
                    else
                        cols=S.subs{2};
                    end
                    if any(rows~=floor(rows)) || any(rows<1) || any(rows>Nrow)
                        error('block circulant: row subscript is not integer or not in range of basic block size')
                    end
                    if any(cols~=floor(cols)) || any(cols<1) || any(cols>Ncol/sym)
                        error('block circulant: column subscript is not integer or not in range of basic block size')
                    end
                    if length(rows)~=size(B.value,1)
                        error('block circulant: rows in B not conformal with A row subscript')
                    end
                    if length(cols)~=size(B.value,2)/sym
                        error('block circulant: columnss in B not conformal with A column subscript')
                    end
                    ix=sum(gen_all_combinations({0:Ncol/sym:Ncol-1,cols(:)'}),2);
                    A.value(rows,ix)=B.value;
                otherwise
                    error('block circulant: only () subscripting allowed');
            end
        end
        %------------------------------------------------------------
        function A=subsref(B,S) %A=B(S)
            Ncol=size(B.value,2);
            Nrow=size(B.value,1);
            sym=B.symmetry;
            switch S.type
                case '()' %extract submatrix using "truncated" indexing
                    if length(S.subs{1})==1 && ischar(S.subs{1}) && S.subs{1}==':'
                        rows=1:Nrow;
                    else
                        rows=S.subs{1};
                    end
                    if length(S.subs{2})==1 && ischar(S.subs{2}) &&S.subs{2}==':'
                        cols=1:Ncol/sym; %only columns in block, not total
                    else
                        cols=S.subs{2};
                    end
                    if any(rows~=floor(rows)) || any(rows<1) || any(rows>Nrow)
                        error('block circulant: row subscript is not integer or not in range of basic block size')
                    end
                    if any(cols~=floor(cols)) || any(cols<1) || any(cols>Ncol/sym)
                        error('block circulant: column subscript is not integer or not in range of basic block size')
                    end
                    ix=sum(gen_all_combinations({0:Ncol/sym:Ncol-1,cols(:)'}),2);
                    a=B.value(rows,ix);
                    A=BlockCirculant(a,sym);
                otherwise
                    error('block circulant: only () subscripting allowed');
            end
        end
        
    end
end

function [y,outDim]=circulant_to2D(mat,rep)
%function y=circulant_to2D(mat,rep) - convert a 3D circulant matrix 
%(rxMxN) to a 2D matrix (MxrN).
%rep is the replication factor k in the description above.
if length(size(mat))==3 %special case for rxMx1 in MATLAB
    [t,rows,cols]=size(mat); %rep should be == t
else
    [rows,cols]=size(mat);
end
y=zeros(rows,rep*cols);
outDim=size(y);
for j=0:rep-1
    y(:,cols*j+[1:cols])=squeeze(mat(j+1,:,:));
end
end
function [y,outDim]=circulant_to3D(mat,rep)
%function y=circulant_to3D(mat,rep) - convert a 2D circulant matrix
% (MxrN)to 3D (rxMxN)).
%rep is the replication factor k in the description above.
[rows,cols]=size(mat);
cols=cols/rep;
y=zeros(rep,rows,cols);
outDim=size(y);
if cols==1
    outDim(3)=1;
end
for i=0:rep-1
    y(i+1,:,:)=mat(:,i*cols+[1:cols]);
end
end
function [y,outDim]=circulant_full(mat,rep)
%function y=circulant_full(mat,rep) - convert a circulant matrix 
% (rxMxN or MxrN) to the full circulant representation (rMxrN).
if length(size(mat))==3
    [t,rows,cols]=size(mat);
else
    [rows,cols]=size(mat);
end
if length(size(mat))<3
    y=zeros(rep*size(mat,1),size(mat,2));
    outDim=size(y);
    for i=0:rep-1
        y(i*size(mat,1)+(1:size(mat,1)),:)=rotate_matrix(mat,-(size(mat,2)/rep)*i,2);
    end
else
    y=zeros(rep*rows,rep*cols);
    outDim=size(y);
    for j=0:rep-1
        for i=0:rep-1
            y(rows*j+[1:rows],cols*mod(i+j,rep)+[1:cols])=squeeze(mat(i+1,:,:));
        end
    end
end
end

function x=circulant_log_det(A,rep)
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
if rep==1 %if no symmetry
    x=det(A);
    return
end
I=sqrt(-1);
N=size(A,1);
[ccA,Dim]=circulant_to3D(A,rep);
A=conj(fft(ccA,[],1));
log_x=0;
for i=0:rep-1
    log_x = log_x + log(det(squeeze(A(i+1,:,:)))); %solve each modified problem
end

x = log_x;
x = real(x);

end

function x=circulant_sqrt(A,rep)
%function x=circulant_invert(A,r) - invert a circulant matrix x=A^-1;
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
if rep==1 %if no symmetry
    x=eye(length(A))/A;
    return
end
I=sqrt(-1);
N=size(A,1);
[ccA,Dim]=circulant_to3D(A,rep);
A=conj(fft(ccA,[],1));
%x=zeros(rep,size(A,2),size(A,3));
x=zeros(rep,size(A,2),size(A,3));
for i=0:rep-1
    x(i+1,:,:)=(sqrt(squeeze(A(i+1,:,:)))).'; %solve each modified problem
end
if N==1
    x=squeeze(ifft(x))';
else
    x=circulant_to2D(ifft(x),rep);
end
x=circulant_transpose(x,rep,1);
end

function x=circulant_invert(A,rep)
%function x=circulant_invert(A,r) - invert a circulant matrix x=A^-1;
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
if rep==1 %if no symmetry
    x=eye(length(A))/A;
    return
end
I=sqrt(-1);
N=size(A,1);
[ccA,Dim]=circulant_to3D(A,rep);
A=conj(fft(ccA,[],1));
%x=zeros(rep,size(A,2),size(A,3));
x=zeros(rep,size(A,2),size(A,3));
for i=0:rep-1
    x(i+1,:,:)=(eye(N)/squeeze(A(i+1,:,:))).'; %solve each modified problem
end
if N==1
    x=squeeze(ifft(x))';
else
    x=circulant_to2D(ifft(x),rep);
end
x=circulant_transpose(x,rep,1);
end

function x=circulant_multiply(A,b,rep)
%function x=circulant_multiply(A,b,rep) - multiply any matrix b by a matrix A with rotational symmetry: x=Ab.
%A is a (M x rN) matrix where r is the order of the symmetry
%b is a (rN by P) matrix of rhs values
%rep=size(A,1)
if rep==1 %if no symmetry
    x=A*b;
    return
end
I=sqrt(-1);
A=conj(fft(circulant_to3D(A,rep),[],1));
b=fft(circulant_to3D(b',rep),[],1);
N=size(b,2);
x=zeros(rep,size(b,2),size(A,2));
for i=0:rep-1
    if N==1 %handle matlab way of treating squeezed vecs
        x(i+1,:,:)=(squeeze(A(i+1,:,:)).'*squeeze(b(i+1,:,:))).'; %solve each modified problem
    else
        x(i+1,:,:)=(squeeze(A(i+1,:,:))*squeeze(b(i+1,:,:)).').'; %solve each modified problem
    end
end

if N==1
    x=squeeze(ifft(x));
else
    if size(A,2)>1
        x=circulant_to2D(ifft(x),rep)';
    else
        x=ifft(x);
    end
end
end

function x=circulant_pinv(A,rep,tol)
%function x=circulant_pinv(A,r)
%invert a circulant matrix x=A^-1;
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
if rep==1 %if no symmetry
    x=eye(length(A))/A;
    return
end
N=size(A,1);
[ccA,Dim]=circulant_to3D(A,rep);
A=conj(fft(ccA,[],1));
M=length(size(A));
x=zeros(Dim);
for i=0:rep-1
    if N==1 %handle matlab way of treating squeezed vecs
        if isempty(tol)
            x(i+1,:,:)=pinv(squeeze(A(i+1,:,:))); %solve each modified problem
        else
            x(i+1,:,:)=pinv(squeeze(A(i+1,:,:)),tol); %solve each modified problem
        end
    else
        if isempty(tol)
            x(i+1,:,:)=(pinv(squeeze(A(i+1,:,:)))).'; %solve each modified problem
        else
            x(i+1,:,:)=(pinv(squeeze(A(i+1,:,:)),tol)).'; %solve each modified problem
        end
    end
end
if Dim(2)==1 && Dim(3)==1
    x=squeeze(ifft(x)');
else
    x=circulant_to2D(ifft(x),rep);
end
x=circulant_transpose(x,rep,1);
end

% %note************************this function has not been thoroughly tested
%  %and is unused by the class BlockCirculant
% function x=circulant_pinv_solve(A,b,rep,e,augmented_rhs)
% %function x=circulant_pinv_solve(A,b,r,e,augmented_rhs) -
% %   solve a problem Ax=b as in circulant_solve with small conditioning matrix added
% %    by concatenating an identity matrix times a small number, e, to approximate the pseudo inverse.
% %    The diagonal of the concatenated matrix is = e*rN*max(A).
% %A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry.
% %b is a (M by P) matrix of rhs values.
% %if augment_rhs is specified, the right hand size is augmented by the specified column vector times the condition number
% %   to force the solution to the rhs is e->Inf. This is like a pseudo inverse with a "push" toward a desired solution.
% %   In this case, e->0 gives the solution A\b and e->Inf gives as a solution the augmented_rhs matrix.
% %   augmented_rhs must be rNx1 or rNxP.
% Am=max(max(abs(A)));
% [M,N]=size(A);
% if size(b,1)~=M*rep
%     error('A and b must conform on inner dimension');
% end
% P=size(b,2);
% if ~exist('augmented_rhs','var')
%     augmented_rhs=zeros(N,P);
% else
%     if size(augmented_rhs,1)~=N
%         error('augmented rhs must be Nx1 or NxP, where N=size(A,2)');
%     end
%     if size(augmented_rhs,2)~=1 & size(augmented_rhs,2)~=P
%         error('size(augmented_rhs,2) must be 1 or size(b,2)');
%     end
%     if size(augmented_rhs,2)<P
%         augmented_rhs=repmat(augmented_rhs,1,P);
%     end
% end
% if rep==1 %if no symmetry
%     x=[A;eye(N)*Am*e]\[b;augmented_rhs];
%     return
% end
% A=[A;[eye(N/rep)*Am*N*e,zeros(N/rep,N-N/rep)]];
% b=reshape(b,[M,rep,P]);
% b=cat(1,b,reshape(augmented_rhs,[N/rep,rep,P]));
% b=reshape(b,M*rep+N,P);
% x=circulant_solve(A,b,rep);
% end

function x=circulant_transpose(A,rep,H)
%function At=circulant_transpose(A,r,H) - transpose a circulant matrix At=A'.
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
%If H is specified, the transpose is Hermitian('), otherwise not (.').
if ~exist('H','var')
    H=0;
else
    H=1;
end
if rep==1
    if H
        x=A';
    else
        x=A.';
    end
    return
end
As=size(A);
M=As(1);
r=rep;
N=As(2)/r;
if N-floor(N)~=0
    error('not valid circulant matrix - dimension 2 must by a multiple of r');
end
A=reshape(A,[M N r]);
A=permute(A,[2 1 3]);
if H
    A=conj(A);
end
x=A(:,:,[1 end:-1:2]);
x=reshape(x,[N M*r]);
return

x=zeros(N,r*M);
if H
    x(:,1:M)=A(1:M,1:N)';
    for i=2:r
        x(:,(i-1)*M+[1:M])=A(:,(r+1-i)*M+[1:M])';
    end
else
    x(:,1:M)=A(1:M,1:N).';
    for i=2:r
        x(:,(i-1)*M+[1:M])=A(:,(r+1-i)*M+[1:M]).';
    end
end
end

function out=rotate_matrix(in,len,dim)
if (nargin<3)
   dim=1;
end
if dim==1
	in=in';
end
s=size(in);
if length(len)==1 %rotate by scalar
   s=s(2);
   s=1+mod(len-1+[1:s],s);
   out=in(:,s);
else %rotate by vector
   if s(1)~=length(len)
      error('rotate-by vector must be same length as matrix dimension');
   end
   s=s(2);
   out=zeros(size(in));
   for i=1:length(len)
      s1=1+mod(len(i)-1+[1:s],s);
      out(i,:)=in(i,s1);
   end
end
if dim==1
	out=out';
end
end

function x=circulant_solve(A,b,rep)
%function x=circulant_solve(A,b,r) - solve a problem in which the matrix A has rotational symmetry: Ax=b.
%A is an (M x rN) matrix, shorthand for rMxrN circulant matrix where r is the order of the symmetry
%b is a (M by P) matrix of rhs values
if rep==1 %if no symmetry
    x=A\b;
    return
end
I=sqrt(-1);
A=conj(fft(circulant_to3D(A,rep),[],1));
b=fft(circulant_to3D(b',rep),[],1);
N=size(b,2);
M=length(size(A));
x=zeros(rep,size(b,2),size(A,3));
for i=0:rep-1
    if M==2 %if A is small (Nx1 is the basic block rather than NxM), treat case where the modified A is 2-D
        if N==1 %handle matlab way of treating squeezed vecs
            x(i+1,:,:)=(A(i+1,:).'\squeeze(b(i+1,:,:))).'; %solve each modified problem
        else
            x(i+1,:,:)=(A(i+1,:).'\squeeze(b(i+1,:,:)).').'; %solve each modified problem
        end
    else %M=3, A is 3-D
        if N==1 %handle matlab way of treating squeezed vecs
            x(i+1,:,:)=(squeeze(A(i+1,:,:))\squeeze(b(i+1,:,:))).'; %solve each modified problem
        else
            x(i+1,:,:)=(squeeze(A(i+1,:,:))\squeeze(b(i+1,:,:)).').'; %solve each modified problem
        end
    end
end
if N==1
    x=squeeze(ifft(x))';
else
    x=circulant_to2D(ifft(x),rep)';
end
end

function out=gen_all_combinations(X,altord)
if ~exist('altord','var')
    altord=0;
end
if altord
    X=fliplr(X);
end
if length(X)==0
    out=[];
    return
end
out=X{end}(:);
for i=length(X)-1:-1:1
    out=[flatten(repmat(X{i}(:)',size(out,1),1)),repmat(out,length(X{i}),1)];
end
if altord
    out=fliplr(out);
end
end

function A=flatten(B)
A=B(:);
end