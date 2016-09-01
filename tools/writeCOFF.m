function writeCOFF(v,c,f,filename)

fid = fopen(filename,'w');

fprintf(fid, 'COFF\n');
fprintf(fid, '%d %d 0\n', size(v,2),size(f,2));

for i = 1:size(v,2)
    fprintf(fid,'%f %f %f %d %d %d 255\n',v(1,i),v(2,i),v(3,i),c(1,i),c(2,i),c(3,i));
end

for i = 1:size(f,2)
    fprintf(fid,'3 %d %d %d\n',f(1,i),f(2,i),f(3,i));
end

fclose(fid);
end