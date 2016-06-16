function demo
warning('off');
close all;
% load data
disp('Load data...');
data = load('usps-mnist.mat');
data.src_n = size(data.data.train.source,1);
data.tar_ntra = size(data.data.train.target,1);
data.tar_ntes = size(data.data.test.target,1);
data.dim = size(data.data.train.source,2);
% init parameters
disp('Init parameters...');
%  k,	C,      dim,	sigma,	lamda,	beta,	gamma,	gammaW
P=[1,   100,      100,     30,      100,   1,      1e-3,   1e-4];
param = initParam(P);
% MTLF
disp('Running MTLF...');
[acc,result]=MTLF(param,data);
acc
end
%%
function TestOnce(list)
i=1; cf=1;
alldata(list(i,1),list(i,2),cf).data=loadData(list(i,1),list(i,2),cf);
%  k,	C,      dim,	sigma,	lamda,	beta,	gamma,	gammaW
P=[1,   100,      100,     30,      100,   1,      1e-3,   1e-4];
param = initParam(P);
[acc,result]=MTLF(list(i,1),list(i,2),cf,param,alldata);
acc
% result
end

function param = initParam(P)
% %  k,	C,      dim,	sigma,	lamda,	beta,	gamma,	gammaW
P=[1,   100,    11,     1,      10000,   1,      1e-4,   1e-5];
param.k=P(1);
param.num_constraints=P(2);
param.dim=P(3);
param.sigma = P(4);             
param.lamda=P(5);               
param.beta = P(6);
param.gamma=P(7);               
param.gammaW=P(8);
param.a=5;
param.b=95;
param.eplsion=1e-7;             
end

function [acc,result]=MTLF(param,data)
% disp('Running CDML');
%% paramter
a=param.a; b=param.b;
num_constraints=param.num_constraints;
k=param.k;dim=param.dim;
%% load data
X = [data.data.train.source;data.data.train.target];
y = [data.labels.train.source';data.labels.train.target'];
Xtest = data.data.test.target;
ytest = data.labels.test.target';
%% pca data dim reduce
Xr = [X;Xtest];
reduced_X=PCA_reduce(Xr,dim);
%% normalization
% [reduced_X,PS]=mapminmax(reduced_X',-1,1);
% reduced_X=reduced_X';
% X=reduced_X(1:size(X,1),:);
% Xtest=reduced_X(size(X,1)+1:size(Xr),:);
% X=reduced_X(1:size(X,1),:);
% Xtest=reduced_X(size(X,1)+1:size(Xr),:);
% [X,minp,maxp,Xtest,mint,maxt] = premnmx(X,Xtest);
%% Compute weights
% disp('Compute weights');
ntra_s = size(data.data.train.source);
ntra_t = size(data.data.train.target);
x_source = X(1:ntra_s,:);
x_target = [X(ntra_s+1:ntra_s+ntra_t,:);Xtest];
[PE, wh_x_source,wh_x_re]=RuLSIF(x_target',x_source');
wh_x_target = zeros(1,size(Xtest,1))+1;
wh_x_source = [wh_x_source,wh_x_target];
%% Compute distance extremes
% disp('Compute distance extremes');
[l, u] = ComputeDistanceExtremes(X, a, b);
% Generate constraint point pairs
% disp('Generate constraint point pairs');
C = GetConstraints(y, num_constraints, l, u);
Xci = X(C(:,1),:); yci = y(C(:,1),:);
Xcj = X(C(:,2),:); ycj = y(C(:,2),:);
%% Optimization
% disp('Optimization');
d=size(X,2); p=num_constraints;
w0 = wh_x_source;
sd_tra = X(1:ntra_s,:);
td_tra = X(ntra_s+1:ntra_s+ntra_t,:);
[A,result]=optimization(C,w0',Xci,Xcj,param,sd_tra,td_tra);
%% Predict
preds = KNN(y, X, A, k, Xtest);
acc = sum(preds==ytest)/size(ytest,1);
acc=acc(1,1);
end

%% optimization
function [At,result]=optimization(C,w0,Xci,Xcj,param,sd_tra,td_tra)
eplsion = param.eplsion;         
sigma = param.sigma;             
lamda=param.lamda; beta = param.beta;     
gamma=param.gamma; gammaW=param.gammaW/sigma;    
A0 = eye(size(Xci,2),size(Xci,2));
E=A0;
At = A0;
ns = size(sd_tra,1);    nt = size(td_tra,1);
e = zeros(ns+nt,1);     e(1:ns,:) = 1;
w0 = w0(1:ns+nt,:); wt = w0;
iter=0;    convA=10000;
% tic
tic
while convA>eplsion && iter<100
    %% updata A
    sumA = zeros(size(Xci,2),size(Xci,2));
    pair_weights=wt(C(:,1),:).*wt(C(:,2),:).*C(:,3);
    for i=1:size(Xci,1)
        vij=Xci(i,:)-Xcj(i,:);
        sumA = sumA+A0*vij'*vij*pair_weights(i)*C(i,3);
    end
    At = At-gamma*(beta*sumA+2*A0);
    %% updata omega
    zeta = zeros(ns+nt,1);
    for k=1:size(Xci,1)
        i = C(k,1); j=C(k,2); deta_ij = C(k,3);
        vij=Xci(k,:)-Xcj(k,:);  vijA = vij*At'; dij=vijA*vijA';
        zeta(i,1) = zeta(i,1)+wt(j)*dij*deta_ij;
        zeta(j,1) = zeta(j,1)+wt(i)*dij*deta_ij;
    end
    xi= sign(max(0,-wt));
    dev1 = 2*lamda*(wt-w0);
    dev2 = beta*zeta;
    dev3 = sigma*(2*(wt'*e-ns)*e+wt.*wt.*xi.*e);
    %     w_dev = 2*lamda*(wt-w0) + beta*zeta + sigma*(2*(wt'*e-ns)*e+wt.*wt.*xi.*e);
    w_dev = dev1+dev2+dev3;
    wt = wt-gammaW*w_dev;     wt(ns+1:ns+nt,:) = 1;
    % wt = w0;
    %% compute threshold
    convA = norm(At-A0);
    convW = norm(wt-w0);
    A0=At;iter=iter+1;
    %
    sumA = 0;
    for k=1:size(Xci,1)
        i = C(k,1); j=C(k,2); deta_ij = C(k,3);
        vij=Xci(k,:)-Xcj(k,:);  vijA = vij*At'; dij=vijA*vijA';
        sumA = sumA + wt(i)*wt(j)*dij*deta_ij;
    end
    f1(iter)=trace(At'*At);
    f2(iter)=lamda*norm(wt-w0)^2;
    f3(iter)=beta*sumA;
    f=f1(iter)+f2(iter)+f3(iter);
    convWs(iter) = convW;   
    convAs(iter) = convA;
    As(iter).A = At;
    ws(iter,:) = wt;
    fs(iter)=f;
    if iter>=2
        fd(iter-1)=fs(iter)-fs(iter-1);
        Ad(iter-1)=norm(As(iter).A -As(iter-1).A);
        wd(iter-1)=norm(ws(iter,:)-ws(iter-1,:));
    else
        fd(1)=1;
        Ad(1)=1;
        wd(1)=1;
    end
end
time = toc;
result.At = At;
result.wt = wt;
result.w0 = w0;
result.Ad = Ad;
result.wd = wd;
result.convAs = convAs;
result.convWs = convWs;
result.fs = fs;
result.fd = fd;
result.f1 = f1;
result.f2 = f2;
result.f3 = f3;
result.time = time;
end
