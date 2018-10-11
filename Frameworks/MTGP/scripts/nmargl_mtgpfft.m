function [nl, gradnl] = nmargl_mtgpfft(logtheta, logtheta_all, covfunc_x, x, y,...
				      m, irank, nx, ind_kf, ind_kx, deriv_range)
% Marginal likelihood and its gradients for multi-task Gaussian Processes
% 
% [nl, gradnl] = nmargl_mtgp(logtheta, logtheta_all, covfunc_x, x, y,...
%                	      m, irank, nx, ind_kf, ind_kx, deriv_range)
%
% To be used in conjunction with Carl Rasmussen's minimize function
% and the gpml package http://www.gaussianprocess.org/gpml/code/matlab/doc/
%
% nl = nmargl_mtgp(logtheta, ...) Returns the marginal negative log-likelihood
% [nl gradnl] =  nmargl_mtgp(logtheta, ...) Returns the gradients wrt logtheta
%
% logtheta    : Column vector of initial values of parameters to be optimized
% logtheta_all: Vector of all parameters: [theta_lf; theta_x; sigma_l]
%                - theta_lf: the parameter vector of the
%                   cholesky decomposition of k_f
%                - theta_x: the parameters of K^x
%                - sigma_l: The log of the noise std deviations for each task
% covfunc_x   : Name of covariance function on input space x
% x           : Unique input points on all tasks 
% y           : Vector of target values
% m           : The number of tasks
% irank       : The rank of K^f 
% nx          : number of times each element of y has been observed 
%                usually nx(i)=1 unless the corresponding y is an average
% ind_kx      : Vector containing the indexes of the data-points in x
%                which each observation y corresponds to
% ind_kf      : Vector containing the indexes of the task to which
%                each observation y corresponds to
% deriv_range : The indices of the parameters in logtheta_all
%                to which each element in logtheta corresponds to
%

% Author: Edwin V. Bonilla
% Last update: 23/01/2011

% *** General settings here ****
config = get_mtgp_config();
MIN_NOISE = config.MIN_NOISE;
% ******************************

if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

D = size(x,2); %#ok  Dimensionality to be used by call to covfunc_x
n = length(y); 

logtheta_all(deriv_range) = logtheta;
ltheta_x = eval(feval(covfunc_x{:}));     % number of parameters for input covariance

nlf = irank*(2*m - irank +1)/2;        % number of parameters for Lf
vlf = logtheta_all(1:nlf);             % parameters for Lf

theta_lf = vlf;
Lf = vec2lowtri_inchol(theta_lf,m,irank);
Kf = Lf*Lf';

theta_x = logtheta_all(nlf+1:nlf+ltheta_x);                     % cov_x parameters
sigma2n = exp(2*logtheta_all(nlf+ltheta_x+1:end));              % Noise parameters
Sigma2n = spdiags(sigma2n, 0, size(Kf, 1), size(Kf, 1));        % Noise Matrix

len_nx = length(nx);
Var_nx = spdiags(1./nx, 0, len_nx, len_nx);

Kx = feval(covfunc_x{:}, theta_x, x);

% Compute Circulant Embedding

% tic;
K_feat = Kf;
K_time = Kx(1 : size(Kx, 1));

% K_feat(abs(K_feat) <= eps) = 0;
% K_time(abs(K_time) <= eps) = 0;

K_time = sparse(K_time);

K_block = kron(K_time, K_feat) + [MIN_NOISE * eye(size(Kf, 1)) + diag(sigma2n), zeros(size(Kf, 1), size(Kf, 1) * (size(Kx, 1) - 1))];

K_circ_det = BlockCirculant(K_block, size(K_block, 2) / size(Kf, 1));

K_block_rev = kron(fliplr(K_time(2 : end-1)), K_feat);
K_block = [K_block, K_block_rev];

K_circulant = BlockCirculant(K_block, size(K_block, 2) / size(Kf, 1));

y_reshape = reshape(y, size(Kx, 1), size(Kf, 1));
y_reshape = y_reshape';
y_reshape = y_reshape(:);

alpha = K_circulant \ [y_reshape; zeros(size(Kf, 1) * (size(Kx, 1)-2), 1)];

alpha = alpha(:, 1 : size(Kx, 1));

alpha_vec = alpha;
alpha_vec = alpha_vec(:);

% K_circ_det still not correct
nl = 0.5*y_reshape'*alpha_vec + 0.5*sum(log_det(K_circ_det)) + 0.5*n*log(2*pi);
% toc;

if (nargout == 2)                      % If requested, its partial derivatives
% 	tic;
  gradnl = zeros(size(logtheta));        % set the size of the derivative vector
	dK_blocks = cell(length(deriv_range), 1);
	
	inv_K = inv(K_circulant);
	inv_K_block = submatrix(inv_K, 1 : size(Kf, 1), 1 : size(Kf, 1) * (2*size(Kx, 1) - 2));
% 	toc;
% 	tic;
  for zz = 1 : length(deriv_range)
		z = deriv_range(zz);
    if (z <= nlf)                          % Gradient wrt  Kf
      [o, p] = pos2ind_tri_inchol(z,m,irank); % determines row and column
			
			Val1 = zeros(m, m);
			Val1(:, o) = Lf(:, p);
			Val = Val1 + Val1';
			Val = sparse(Val);
			dK_block = [kron(K_time, Val), kron(fliplr(K_time(2 : end-1)), Val)];
    elseif ( z <= (nlf+ltheta_x) )           % Gradient wrt parameters of Kx
      z_x =  z - nlf;
      dKx = feval(covfunc_x{:},theta_x, x, z_x);
			dKx = dKx(1 : size(Kx, 1));
			dKx = sparse(dKx);
			dK_block = [kron(dKx, Kf), kron(fliplr(dKx(:, 2 : end-1)), Kf)];
    elseif ( z >= (nlf+ltheta_x+1) )         % Gradient wrt Noise variances
      Val = zeros(m,m);
      kk = z - nlf - ltheta_x;
      Val(kk,kk) = 2*Sigma2n(kk,kk);
			Val = sparse(Val);
			
			ind_K_feat = kron(ones(size(Kx, 1), 1), transpose(1 : size(Kf, 1)));
      dK_block = Val(ind_K_feat, ind_K_feat) .* Var_nx;
			% Multiplication is wrong
    end % endif z
		
		dK_block = sparse(dK_block);
		dK_blocks{zz} = dK_block;
  end % end for derivatives
% 	toc;
	
% 	tic;
	alpha_dKs_sum = cell(length(deriv_range), 1);
	for zz = 1 : length(deriv_range)
		dK_block = dK_blocks{zz};
% 		[is, js, dK] = find(dK_block);
		if nnz(dK_block) < 1000
			[is, js, dK] = find(dK_block);
		else
			[is1, js1, dK1] = find(dK_block, 500);
			[is2, js2, dK2] = find(dK_block, 500, 'last');

			is = [is1, is2];
			js = [js1, js2];
			dK = [dK1, dK2];
		end
		
		part_sum = 0;
		z = deriv_range(zz);
		if z < (nlf+ltheta_x+1)
			for i = 1 : size(Kx, 1)
				js_curr = mod(js + (i-1) * size(Kf, 1) - 1, size(dK_block, 2)) + 1;
				js_ind = (js_curr > size(Kf, 1) * size(Kx, 1));
				
				js_curr(js_ind) = [];
				is_curr = is(~js_ind);
				dK_curr = dK(~js_ind);
				
% 				dK_block_curr = dK_block(:, mod( -(i-1)*size(Kf, 1) : -(i-1) * size(Kf, 1) + size(Kf, 1) * size(Kx, 1) - 1, size(dK_block, 2)) + 1);
				for index = 1 : length(is_curr)
					part_sum = part_sum + dK_curr(index) * alpha_vec(is_curr(index) + (i-1) * size(Kf,1)) * alpha_vec(js_curr(index));
				end
% 				part_sum = part_sum + sum(sum(dK_block_curr .* (alpha_vec(1 + (i-1) * size(Kf,1) : i * size(Kf,1)) * alpha_vec')));
			end
			alpha_dKs_sum{zz} = part_sum;
		else
			for index = 1 : length(is)
				part_sum = part_sum + dK(index) * alpha_vec(is(index)) * alpha_vec(js(index));
			end
			alpha_dKs_sum{zz} = part_sum;
		end
	end
% 	toc;

% 	tic;
	count = 1;
	for zz = 1 : length(deriv_range)
		dK_block = dK_blocks{zz};
% 		[is, js, dK] = find(dK_block);
		if nnz(dK_block) < 1000
			[is, js, dK] = find(dK_block);
		else
			[is1, js1, dK1] = find(dK_block, 500);
			[is2, js2, dK2] = find(dK_block, 500, 'last');

			is = [is1, is2];
			js = [js1, js2];
			dK = [dK1, dK2];
		end
		
		z = deriv_range(zz);
		part_sum = 0;
		if z < (nlf+ltheta_x+1)
			for i = 1 : size(Kx, 1)
				js_curr = mod(js + (i-1) * size(Kf, 1) - 1, size(dK_block, 2)) + 1;
				js_ind = (js_curr > size(Kf, 1) * size(Kx, 1));
				
				js_curr(js_ind) = [];
				is_curr = is(~js_ind);
				dK_curr = dK(~js_ind);
				
% 				inv_K_block_curr = inv_K_block(:, mod( -(i-1)*size(Kf, 1) : -(i-1) * size(Kf, 1) + size(Kf, 1) * size(Kx, 1) - 1, size(inv_K_block, 2)) + 1);
% 				dK_block_curr = dK_block(:, mod( -(i-1)*size(Kf, 1) : -(i-1) * size(Kf, 1) + size(Kf, 1) * size(Kx, 1) - 1, size(dK_block, 2)) + 1);

				for index = 1 : length(is_curr)
					part_sum = part_sum + dK_curr(index) * inv_K_block(is_curr(index), mod(js_curr(index) - (i-1) * size(Kf, 1) - 1, size(dK_block, 2)) + 1);
				end
% 				part_sum = part_sum + sum(sum(inv_K_block_curr .* dK_block_curr));
			end
		else
			is_curr = mod(is - 1, size(Kf, 1)) + 1;
			js_curr = mod(js - floor((is-1)/size(Kf, 1)) * size(Kf, 1), size(inv_K_block, 2));
			for index = 1 : length(is_curr)
				part_sum = part_sum + dK(index) * inv_K_block(is_curr(index), js_curr(index));
			end

		end
		gradnl(count) = part_sum - alpha_dKs_sum{zz};
		count = count + 1;
	end
	gradnl = gradnl/2;
% 	toc;
end % end if nargout ==2
