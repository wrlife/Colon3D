clear;

[v,c,f] = readCOFF('temp1.off');
f = f + 1;

missing = sum(c);
idx = find(missing == 0);

i1 = ismember(f(1,:),idx);
i2 = ismember(f(2,:),idx);
i3 = ismember(f(3,:),idx);

fidx = find((i1+i2+i3) == 0);
f = f(:,fidx) - 1;
writeCOFF(v,c,f,'fused.off');