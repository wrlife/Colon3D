function writeOFF(v,f,filename)

fid = fopen(filename,'w');

fprintf(fid, 'OFF\n');
fprintf(fid, '%d %d 0\n', size(v,2),size(f,2));

for i = 1:size(v,2)
    fprintf(fid,'%f %f %f\n',v(1,i),v(2,i),v(3,i));
end

for i = 1:size(f,2)
    fprintf(fid,'3 %d %d %d\n',f(1,i),f(2,i),f(3,i));
end

fclose(fid);
end