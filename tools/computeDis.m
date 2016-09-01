function error = computeDis(v,f,vo,fo)

error = zeros(size(v,2),1);
for i = 1:size(v,2)
        dif = repmat(v(:,i),1,size(vo,2)) - vo;
        dif = sum(dif.*dif);
        idx = find(dif == min(dif)); idx = idx(1);
        [idxI,idxJ] = find(fo == idx); 
    
        minDis = 1000;
        for j = 1:size(idxI,1) % find the closest triangle
            TRI = [vo(:,fo(1,idxJ(j)))';...
                   vo(:,fo(2,idxJ(j)))';...
                   vo(:,fo(3,idxJ(j)))'];
               
            [dis,PP0] = pointTriangleDistance(TRI,v(:,i)');
            
            if (dis < minDis)
               minDis = dis;
               error(i) = dis;
            end
        end
end

end