function [dis,PP0,r1,r2,r3] = pointTriangleDistanceNd(TRI,P0)
%     H = 2*(TRI*TRI');
%     f = -2*P0*TRI';
    C = TRI';
    d = P0';
    Aeq = [1,1,1];
    beq = 1;
    
    %options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
    %[x,fval] = quadprog(H,f,[],[],Aeq,beq,zeros(3,1),[],[],options);
    
    options = optimoptions('lsqlin','Algorithm','active-set','Display','off');
    x = lsqlin(C,d,[],[],Aeq,beq,zeros(3,1),[],[],options);
    
    PP0 = (TRI'*x)';
    dis = norm(PP0-P0);

    r1 = x(1);
    r2 = x(2);
    r3 = x(3);
end