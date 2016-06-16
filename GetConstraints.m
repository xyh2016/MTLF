function C = GetConstraints(y, num_constraints, l, u)
% C = GetConstraints(y, num_constraints, l, u)
%
% Get ITML constraint matrix from true labels.  See ItmlAlg.m for
% description of the constraint matrix format

m = length(y);
C = zeros(num_constraints, 4);

for (k=1:num_constraints),
    i = ceil(rand * m);
    j = ceil(rand * m);
    if (y(i) == y(j)),
        C(k,:) = [i j 1 l];
    else
        C(k,:) = [i j -1 u];
    end
end
index1 = find(C(:,3)==1);
index2 = find(C(:,3)==-1);
C(index2,3) = -size(index1,1)/size(index2,1);
% C(index2,3) = 0;
end


