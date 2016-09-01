function cp2 = transferPoints(p,rawCT,deformedCT)
    [v_raw,f_raw] = readOFF(rawCT);
    [v_deformed,f_deformed] = readOFF(deformedCT);
    
    f_raw = f_raw + 1;
    f_deformed = f_deformed + 1;
    
    cp1 = zeros(3,size(p,2));
    cp2 = zeros(3,size(p,2));
    
    for i = 1:size(p,2)
        dif(1,:) = v_raw(1,:) - p(1,i);
        dif(2,:) = v_raw(2,:) - p(2,i);
        dif(3,:) = v_raw(3,:) - p(3,i);
    
        [M,I] = min(sum(dif.*dif)); % find the closest point
    
        [idxI,idxJ] = find(f_raw == I); 
    
        minDis = 1000;
        for j = 1:size(idxI,1) % find the closest triangle
            TRI = [v_raw(:,f_raw(1,idxJ(j)))';...
                   v_raw(:,f_raw(2,idxJ(j)))';...
                   v_raw(:,f_raw(3,idxJ(j)))'];
               
            [dis,PP0] = pointTriangleDistance(TRI,p(:,i)');
            
            if (dis < minDis)
                cp1(:,i) = PP0';
                [r1,r2,r3] = triangleInterpolation(TRI(1,:),TRI(2,:),TRI(3,:),PP0);
                cp2(:,i) = v_deformed(:,f_raw(1,idxJ(j)))*r1 + ...
                           v_deformed(:,f_raw(2,idxJ(j)))*r2 +...
                           v_deformed(:,f_raw(3,idxJ(j)))*r3;
            end
        end
    end

end