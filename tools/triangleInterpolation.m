function [r1,r2,r3] = triangleInterpolation(a,b,c,p)
    a1 = triangleArea3d(p, b, c);
    a2 = triangleArea3d(p, c, a);
    a3 = triangleArea3d(p, a, b);
    
    r1 = a1 / (a1+a2+a3);
    r2 = a2 / (a1+a2+a3);
    r3 = a3 / (a1+a2+a3);
end

