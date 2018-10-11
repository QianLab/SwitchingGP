function C = matern(d, range, nu)

alpha = sqrt(2*nu) .* d ./ range;
C = 2^(1-nu) .* alpha.^nu .* besselk(nu, alpha);

end