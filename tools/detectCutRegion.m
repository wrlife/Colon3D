function R = detectCutRegion(X,Y,Z,tangencyThreshold)

N = size(X,2);
M = size(X,1);

%idx = vec2mat([1:S],N);
%idx = idx(2:M,1:N-1);
%idxU = idx - 720;
%idxD = idx - 720 + 1;
%idxR = idx + 1;

DX1 = X(1:M-1,1:N-1)-X(1:M-1,2:N);
DY1 = Y(1:M-1,1:N-1)-Y(1:M-1,2:N);
DZ1 = Z(1:M-1,1:N-1)-Z(1:M-1,2:N);

DX2 = X(1:M-1,1:N-1)-X(2:M,1:N-1);
DY2 = Y(1:M-1,1:N-1)-Y(2:M,1:N-1);
DZ2 = Z(1:M-1,1:N-1)-Z(2:M,1:N-1);

DX3 = X(1:M-1,1:N-1)-X(2:M,2:N);
DY3 = Y(1:M-1,1:N-1)-Y(2:M,2:N);
DZ3 = Z(1:M-1,1:N-1)-Z(2:M,2:N);

dif1 = [DX1(:)'; DY1(:)'; DZ1(:)'];
dif2 = [DX2(:)'; DY2(:)'; DZ2(:)'];
dif3 = [DX3(:)'; DY3(:)'; DZ3(:)'];

XP = X(1:M-1,1:N-1);
YP = Y(1:M-1,1:N-1);
ZP = Z(1:M-1,1:N-1);
V =    [XP(:)';YP(:)';ZP(:)'];

%% COMPUTE TANGENCY
normal1 = cross(dif1,dif2);
normal2 = cross(dif2,dif3);

nl1 = sqrt(sum(normal1.*normal1));
nl2 = sqrt(sum(normal2.*normal2));
nl1 = repmat(nl1,3,1);
nl2 = repmat(nl2,3,1);
normal1 = normal1 ./ nl1;
normal2 = normal2 ./ nl2;

l = sqrt(sum(V.*V));
l = repmat(l,3,1);
tangency1 = abs(sum(normal1.*(V./l)));
tangency2 = abs(sum(normal2.*(V./l)));

o = (tangency1<tangencyThreshold)|(tangency2<tangencyThreshold);

R = ones(M,N); 
R(1:M-1,1:N-1) = vec2mat(o,M-1)';


%%
% dif1 = sqrt(sum(dif1.*dif1));
% dif2 = sqrt(sum(dif2.*dif2));
% dif3 = sqrt(sum(dif3.*dif3));
% 
% threshold2 = 10;
% o = (dif1>threshold2)     |(dif2>threshold2)     |(dif3>threshold2);

%% 
% occludedIdx = idx(o);
% fidx1 = ismember(f(1,:),occludedIdx);
% fidx2 = ismember(f(2,:),occludedIdx);
% fidx3 = ismember(f(3,:),occludedIdx);
% fidx = ~fidx1 & ~fidx2 & ~fidx3;
% f = f(:,fidx);

%% output
% c(:,occludedIdx) = repmat([0;255;0],1,size(occludedIdx,2));
% writeCOFF(v,c,f-1,'temp.off');
% tangency = 1./(1+exp(-50*(tangency-0.15)));
% c = zeros(3,size(v,2)) + 255;
% c(1,idx) = int16(tangency*255);
%writeCOFF(v,c,f-1,'temp.off');

end


