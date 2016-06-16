function x=mylinsolve(A,b)

%Solve Ax=b (A and b are both dense)

%tic
%x=A\b;
%toc
%tic

sflag = issparse(A);

if sflag == 1
    R = cholinc(A,10^(-10));
    x = R\(R'\b);
    x = full(x);
else
    R = chol(A);
    x = R\(R'\b);
end
%toc

%%There seem to be a lot of functions to solve Ax=b...

%x=gmres(A,b);
%x=bicg(A,b);
%x=bicgstab(A,b);
%x=cgs(A,b);
%x=minres(A,b);
%x=pcg(A,b);
%x=qmr(A,b);
%x=symmlq(A,b);
%opts.SYM=true;x=linsolve(A,b,opts);
%x=inv(A)*b;
