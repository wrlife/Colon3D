clear;
addpath('tools');

[v,f] = readOFF('registered2815_cut.off');
[v1,f1] = readOFF('fusion_seq1_seq2_cut.off');
[v2,f2] = readOFF('temp9.off');
%[v1,f1] = readOFF('registered2815.off');
%[v2,f2] = readOFF('origin2815.off');

f1 = f1 + 1;
n = size(v,2);
n1 = size(v1,2);
    
v_new = zeros(size(v));
for i = 1:n
    dif = repmat(v(:,i),1,n1) - v1;
    dif = sum(dif.*dif);
    idx = find(dif == min(dif)); idx = idx(1);
    [idxI,idxJ] = find(f1 == idx); 
    
    minDis = 1000;
    for j = 1:size(idxI,1) % find the closest triangle
    	TRI = [v1(:,f1(1,idxJ(j)))';...
               v1(:,f1(2,idxJ(j)))';...
               v1(:,f1(3,idxJ(j)))'];
               
        [dis,PP0] = pointTriangleDistance(TRI,v(:,i)');
            
        if (dis < minDis)
                [r1,r2,r3] = triangleInterpolation(TRI(1,:),TRI(2,:),TRI(3,:),PP0);
                r = v2(:,f1(1,idxJ(j)))'*r1+...
                    v2(:,f1(2,idxJ(j)))'*r2+...
                    v2(:,f1(3,idxJ(j)))'*r3;
                minDis = dis;
        end
    end
    
    v_new(:,i) = r;
    
    if (mod(i,1000) == 0)
        fprintf('transferring: %d\n',i);
    end
end

writeOFF(v_new,f,'registered2815.off');