clear;

addpath('tools');
tc = load('tc1875.txt');
[vc,fc] = readOFF('registered1875.off');
[vd,fd] = readOFF('registered1875.off');
[vo,fo] = readOFF('origin1875.off');

nc = size(vc,2);
nd = size(vd,2);
nr = size(vo,2);

tfinal = zeros(nc,2);
for i = 1:nc    
    dif = repmat(vc(:,i),1,nd) - vd; 
    dif = sum(dif.*dif);
    
    idx = find(dif == min(dif));
	idx = idx(1);
    
    tfinal(i,:) = tc(idx,:);
end

save tfinal.txt tc -ascii;

fid = fopen('temp.obj','w');

fprintf(fid,'####\n');
fprintf(fid,'#\n');
fprintf(fid,'# OBJ File Generated by Meshlab\n');
fprintf(fid,'#\n');
fprintf(fid,'####\n');
fprintf(fid,'# Object texture.obj\n');
fprintf(fid,'#\n');
fprintf(fid,'# Vertices: %d\n',size(vc,2));
fprintf(fid,'# Faces: %d\n',size(fc,2));
fprintf(fid,'#\n');
fprintf(fid,'####\n');
fprintf(fid,'mtllib ./temp.obj.mtl\n');

for i = 1:size(vc,2)
    fprintf(fid,'v %f %f %f\n',vc(1,i),vc(2,i),vc(3,i));
end

fprintf(fid,'# %d vertices, 0 vertices normals\n',size(vc,2));
fprintf(fid,'\n');
fprintf(fid,'usemtl material_0\n');

for i = 1:size(vc,2)
    fprintf(fid,'vt %f %f\n',tfinal(i,1),tfinal(i,2));
end
% fprintf(fid,'vt %f %f\n',0.3,0.6);
% fprintf(fid,'vt %f %f\n',0.3,0.9);
% fprintf(fid,'vt %f %f\n',0.6,0.9);

fc = fc+1;
t = fc;

for i = 1:size(fc,2)
	fprintf(fid,'f %d/%d %d/%d %d/%d \n',fc(1,i),t(1,i),fc(2,i),t(2,i),fc(3,i),t(3,i));
end 

fprintf(fid,'# %d faces, %d coords texture\n',size(fc,2),size(vc,2));

fprintf(fid,'# End of File\n');
fclose(fid);