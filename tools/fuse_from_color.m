clear;

addpath('tools');
addpath('DATA');

[v1,c1,f1] = readCOFF('fused_larynx.off');
[v2,c2,f2] = readCOFF('transfered_epiglottis.off');
%[v3,c3,f3] = readCOFF('registered1875_color.off');
[v,f] = readOFF('fused_overall.off');
f1 = f1 + 1;
f2 = f2 + 1;
%f3 = f3 + 1;

n = size(v,2);
n1 = size(v1,2);
n2 = size(v2,2);
%n3 = size(v3,2);

c = zeros(3,n);
for i = 1:n
    dif1 = repmat(v(:,i),1,n1) - v1;
    dif1 = sum(dif1.*dif1);
    idx1 = find(dif1 == min(dif1)); idx1 = idx1(1);
    color1 = c1(:,idx1);        
    euc1 = norm(v(:,i)-v1(:,idx1));
    [idxI,idxJ] = find(f1 == idx1); 
    minDis = 1000;
    for j = 1:size(idxI,1) % find the closest triangle
        TRI = [v1(:,f1(1,idxJ(j)))';...
               v1(:,f1(2,idxJ(j)))';...
               v1(:,f1(3,idxJ(j)))'];
               
        [dis,PP0] = pointTriangleDistance(TRI,v(:,i)');
            
        if (dis < minDis)
            euc1 = dis;
            [r1,r2,r3] = triangleInterpolation(TRI(1,:),TRI(2,:),TRI(3,:),PP0);
            color1 = c1(:,f1(1,idxJ(j)))*r1 + ...
                     c1(:,f1(2,idxJ(j)))*r2 +...
                     c1(:,f1(3,idxJ(j)))*r3;
        end
    end
    
    %euc1 = 4/(1+exp(-10*(euc1-3)));
    euc1 = 1/(euc1 + 0.0001);
    
    %if (euc1<0.5)
        dif2 = repmat(v(:,i),1,n2) - v2;
        dif2 = sum(dif2.*dif2);
        idx2 = find(dif2 == min(dif2)); idx2 = idx2(1);
        color2 = c2(:,idx2);        
        euc2 = norm(v(:,i)-v2(:,idx2));
        [idxI,idxJ] = find(f2 == idx2); 
        minDis = 1000;
        for j = 1:size(idxI,1) % find the closest triangle
            TRI = [v2(:,f2(1,idxJ(j)))';...
                   v2(:,f2(2,idxJ(j)))';...
                   v2(:,f2(3,idxJ(j)))'];
               
            [dis,PP0] = pointTriangleDistance(TRI,v(:,i)');
            
            if (dis < minDis)
                euc2 = dis;
                [r1,r2,r3] = triangleInterpolation(TRI(1,:),TRI(2,:),TRI(3,:),PP0);
                color2 = c2(:,f2(1,idxJ(j)))*r1 + ...
                         c2(:,f2(2,idxJ(j)))*r2 +...
                         c2(:,f2(3,idxJ(j)))*r3;
            end
        end
        euc2 = 1/(euc2+0.0001);
   %%
%         dif3 = repmat(v(:,i),1,n3) - v3;
%         dif3 = sum(dif3.*dif3);
%         idx3 = find(dif3 == min(dif3)); idx3 = idx3(1);
%         color3 = c3(:,idx3);        
%         euc3 = norm(v(:,i)-v3(:,idx3));
%         [idxI,idxJ] = find(f3 == idx3); 
%         minDis = 1000;
%         for j = 1:size(idxI,1) % find the closest triangle
%             TRI = [v3(:,f3(1,idxJ(j)))';...
%                    v3(:,f3(2,idxJ(j)))';...
%                    v3(:,f3(3,idxJ(j)))'];
%                
%             [dis,PP0] = pointTriangleDistance(TRI,v(:,i)');
%             
%             if (dis < minDis)
%                 euc3 = dis;
%                 [r1,r2,r3] = triangleInterpolation(TRI(1,:),TRI(2,:),TRI(3,:),PP0);
%                 color3 = c3(:,f3(1,idxJ(j)))*r1 + ...
%                          c3(:,f3(2,idxJ(j)))*r2 +...
%                          c3(:,f3(3,idxJ(j)))*r3;
%             end
%         end
%         euc3 = 1/(euc3+0.0001);
        euc3 = 0;
        color3 = [0;0;0];

        weight1 = euc1 / (euc1 + euc2 + euc3);
        weight2 = euc2 / (euc1 + euc2 + euc3);
        weight3 = euc3 / (euc1 + euc2 + euc3);
    
        w1 = power(weight1,4);
        w2 = power(weight2,4);
        w3 = power(weight3,4);
        
        weight1 = w1 / (w1 + w2 + w3);
        weight2 = w2 / (w1 + w2 + w3);
        weight3 = w3 / (w1 + w2 + w3);
%     else
%         weight1 = 1;
%         weight2 = 0;
%         color2 = [0;0;0];
%         weight3 = 0;
%         color3 = [0;0;0];
%     end
    
    c(:,i) = (weight1*color1 + weight2*color2 + weight3*color3) / (weight1 + weight2 + weight3);
    if (mod(i,100)==0) 
        fprintf('%d\n',i);
    end
end

c = int16(c);
c(c(:)>255) = 255;
c(c(:)<0) = 0;
writeCOFF(v,c,f,'fused.off');