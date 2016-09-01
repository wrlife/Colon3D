function cp = mapPoints2OFF(p,file)
    [v,f] = readOFF(file);
    f = f+1;
    cp = zeros(size(p));
    
    for i = 1:size(p,2)
        dif(1,:) = v(1,:) - p(1,i);
        dif(2,:) = v(2,:) - p(2,i);
        dif(3,:) = v(3,:) - p(3,i);
    
        [M,I] = min(sum(dif.*dif)); % find the closest point
    
        [idxI,idxJ] = find(f == I); 
    
        minDis = 1000;
        for j = 1:size(idxI,1) % find the closest triangle
            TRI = [v(:,f(1,idxJ(j)))';...
                   v(:,f(2,idxJ(j)))';...
                   v(:,f(3,idxJ(j)))'];
               
            [dis,PP0] = pointTriangleDistance(TRI,p(:,i)');
            
            if (dis < minDis)
                cp(:,i) = PP0';
            end
        end
    end
end