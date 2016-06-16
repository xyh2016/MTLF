function reduced_X=PCA_reduce(X,retain_dimensions)
% X = rand(200,800);
% retain_dimensions = 100;
[U,S,V] = svd(cov(X));
% U = princomp(X);
reduced_X = X*U(:,1:retain_dimensions);
% reduced_X = X*V(:,1:retain_dimensions);
end