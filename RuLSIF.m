function [PE, wh_x_de,wh_x_re]=RuLSIF(x_nu,x_de,x_re, alpha,sigma_list,lambda_list,b,fold)
%
% Relative Unconstrained least-squares importance fitting (with leave-one-out cross validation)
%
% Estimating relative ratio of probability densities
%   \frac{ p_{nu}(x) }{al*p_{nu}(x) + (1 - al)* p_{de}(x) }
% from samples
%    { xde_i | xde_i\in R^{d} }_{i=1}^{n_{de}}
% drawn independently from p_{de}(x) and samples
%    { xnu_i | xnu_i\in R^{d} }_{i=1}^{n_{nu}}
% drawn independently from p_{nu}(x).
%
% Usage:
%       [PE, wh_x_nu,wh_x_de]=RuLSIF(x_nu,x_de,x_re,sigma_list,lambda_list,b)
%
% Input:
%    x_de:         d by n_de sample matrix corresponding to `denominator' (iid from density p_de)
%    x_nu:         d by n_nu sample matrix corresponding to `numerator'   (iid from density p_nu)
%    x_re:         (OPTIONAL) d by n_re reference sample matrix
%    sigma_list:   (OPTIONAL) Gaussian width
%                  If sigma_list is a vector, one of them is selected by cross validation.
%                  If sigma_list is a scalar, this value is used without cross validation.
%                  If sigma_list is empty/undefined, Gaussian width is chosen from
%                  some default canditate list by cross validation.
%    lambda_list: (OPTIONAL) regularization parameter
%                 If lambda_list is a vector, one of them is selected by cross validation.
%                 If lambda_list is a scalar, this value is used without cross validation
%                 If lambda_list is empty, Gaussian width is chosen from
%                 some default canditate list by cross validation
%    b:           (OPTINLAL) positive integer representing the number of kernels (default: 100)
%    fold:        (OPTINLAL) positive integer representing the number of folds
%                 in cross validation / 0: leave-one-out (default: 0)
%
% Output:
%         PE:     theta-relative PEarson divergence
%    wh_x_de:     estimates of density ratio w=p_nu/(al*p_nu + (1 - al)*p_de) at x_de
%    wh_x_re:     estimates of density ratio w=p_nu/(al*p_nu + (1 - al)*p_de) at x_re (if x_re is provided)
%
% (c) Makoto Yamada & Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     yamada@sg.cs.titech.ac.jp, sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/RuLSIF/

seed = 1;
rand('state',seed);
randn('state',seed);

%keyboard
if nargin<2
    error('number of input arguments is not enough!!!')
end

[d,   n_de]=size(x_de);
[d_nu,n_nu]=size(x_nu);
if d~=d_nu
    error('dimension of two samples are diferent!!!')
end

isdisp = 1;
if nargin<3 || isempty(x_re)
    isdisp = 0;
end

if nargin<4 || isempty(alpha)
    alpha = 0.5;
end


if nargin<5 || isempty(sigma_list)
    x = [x_nu x_de];
    %tmp=repmat(sum(x.^2,1),[size(x_nu,2)+size(x_de,2) 1]);
    %dist2=tmp+tmp'-2*x'*x;
    %sigma_list=sqrt(median(dist2(:)))*[0.6 0.8 1 1.2 1.4];
    med = compmedDist(x');
    sigma_list = med*[0.6 0.8 1 1.2 1.4];
elseif sum(sigma_list<=0)>0
    error('Gaussian width must be positive')
end
if nargin<6 || isempty(lambda_list)
    lambda_list=10.^[-3 -2 -1 0 1];
    %lambda_list=logspace(-3,1,9); % Candidates of regularization parameter
elseif sum(lambda_list<0)>0
    error('regularization parameter must be non-negative')
end
if nargin<7 || isempty(b)
    b = 100;
end
if nargin<8 || isempty(fold)
    fold = 5; %leave-one-out
end
%fold
%%%%%%%%%%%%%%%% Choose Gaussian kernel center `x_ce'
rand_index=randperm(n_nu);
b=min(b,n_nu);
x_ce=x_nu(:,rand_index(1:b));
n_min=min(n_de,n_nu);

score_cv=zeros(length(sigma_list),length(lambda_list));

if length(sigma_list)==1 && length(lambda_list)==1 % need cross-validation?
    sigma_chosen=sigma_list;
    lambda_chosen=lambda_list;
else
    %%%%%%%%%%%%%%%% Searching Gaussian kernel width `sigma_chosen'
    %%%%%%%%%%%%%%%% and regularization parameter `lambda_chosen'
    if fold~=0 % k-fold cross-validation
        cv_index_nu=randperm(n_nu);
        cv_split_nu=floor([0:n_nu-1]*fold./n_nu)+1;
        cv_index_de=randperm(n_de);
        cv_split_de=floor([0:n_de-1]*fold./n_de)+1;
    end
    for sigma_index=1:length(sigma_list)
        sigma=sigma_list(sigma_index);
        K_de=kernel_Gaussian(x_de,x_ce,sigma)';
        K_nu=kernel_Gaussian(x_nu,x_ce,sigma)';
        %       if fold==0 % leave-one-out cross-validation
        %         K_de2=K_de(:,1:n_min);
        %         K_nu2=K_nu(:,1:n_min);
        %         H=K_de*K_de'/size(K_de,2);
        %         h=mean(K_nu,2);
        %       end
        score_tmp=zeros(fold,length(lambda_list));
        for k=1:fold
            Ktmp1=K_de(:,cv_index_de(cv_split_de~=k));
            Ktmp2=K_nu(:,cv_index_nu(cv_split_nu~=k));
            
            Ktmp = alpha/size(Ktmp2,2)*Ktmp2*Ktmp2' + (1-alpha)/size(Ktmp1,2)*Ktmp1*Ktmp1';
            mKtmp = mean(K_nu(:,cv_index_nu(cv_split_nu~=k)),2);
            for lambda_index=1:length(lambda_list)
                lambda=lambda_list(lambda_index);
                %         if fold==0 % leave-one-out cross-validation
                %           C=H+lambda*(n_de-1)/n_de*eye(b);
                %           invC=inv(C);
                %           beta=invC*h;
                %           invCK_de=invC*K_de2;
                %           tmp=n_de*ones(1,n_min)-sum(K_de2.*invCK_de,1);
                %           B0=beta*ones(1,n_min)+invCK_de*diag((beta'*K_de2)./tmp);
                %           B1=invC*K_nu2+invCK_de*diag(sum(K_nu2.*invCK_de,1)./tmp);
                %           A=max(0,(n_de-1)/(n_de*(n_nu-1))*(n_nu*B0-B1));
                %           wh_x_de2=sum(K_de2.*A,1)';
                %           wh_x_nu2=sum(K_nu2.*A,1)';
                %           score_cv(sigma_index,lambda_index)=mean(wh_x_de2.^2)/2-mean(wh_x_nu2);
                %         else % k-fold cross-validation
                thetat_cv=mylinsolve( Ktmp +lambda*eye(b), mKtmp);
                thetah_cv=thetat_cv;%max(0,thetat_cv);
                score_tmp(k,lambda_index)= alpha*mean((K_nu(:,cv_index_nu(cv_split_nu==k))'*thetah_cv).^2)/2 ...
                    + (1-alpha)*mean((K_de(:,cv_index_de(cv_split_de==k))'*thetah_cv).^2)/2 ...
                    -mean(K_nu(:,cv_index_nu(cv_split_nu==k))'*thetah_cv);
                % keyboard;
            end % for fold
            %keyboard
            score_cv(sigma_index,:)=mean(score_tmp,1);
            %end % if fold==0
        end % for lambda_index
    end % for sigma_index
    
    [score_cv_tmp,lambda_chosen_index]=min(score_cv,[],2);
    [score,sigma_chosen_index]=min(score_cv_tmp);
    lambda_chosen=lambda_list(lambda_chosen_index(sigma_chosen_index));
    sigma_chosen=sigma_list(sigma_chosen_index);
end %cross-validation
%keyboard;
%%%%%%%%%%%%%%%% Computing the final solution `wh_x_de'
K_de=kernel_Gaussian(x_de,x_ce,sigma_chosen)';
K_nu=kernel_Gaussian(x_nu,x_ce,sigma_chosen)';

thetat=mylinsolve(alpha*K_nu*K_nu'/n_nu + (1-alpha)*K_de*K_de'/n_de  +lambda_chosen*eye(b),mean(K_nu,2));
thetah=thetat;%max(0,thetat);
wh_x_de=(K_de'*thetah)';
wh_x_nu=(K_nu'*thetah)';

if isdisp ~= 0
    K_di=kernel_Gaussian(x_re,x_ce,sigma_chosen)';
    wh_x_re=(K_di'*thetah)';
else
    wh_x_re = 0;
end
% thetah=max(0,thetat);
PE = mean(wh_x_nu) - 1/2*(alpha*mean(wh_x_nu.^2) + (1-alpha)*mean(wh_x_de.^2)) - 1/2;

wh_x_de = max(0,wh_x_de);
wh_x_re = max(0,wh_x_re);

