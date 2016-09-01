function isBoundary = findBoundary(v,f)
    edgeTable = table([f(1,:)';f(2,:)';f(3,:)'],[f(2,:)';f(3,:)';f(1,:)']);
    isBoundary = zeros(1,size(v,2));
    
    for i = 1:size(f,2)*3
        t = table(edgeTable.Var2(i),edgeTable.Var1(i));
        if (~ismember(t,edgeTable))
            isBoundary(edgeTable.Var1(i)) = 1;
            isBoundary(edgeTable.Var2(i)) = 1;
        end
        
        if (mod(i,100) == 0)
            fprintf('isBoundary: %d\n',i);
        end
    end
end