function px=pdf_Gaussian(x,mu,sigma)
  
  [d,nx]=size(x);

  tmp=(x-repmat(mu,[1 nx]))./repmat(sigma,[1 nx])/sqrt(2);
  px=(2*pi)^(-d/2)/prod(sigma)*exp(-sum(tmp.^2,1));
