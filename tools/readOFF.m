function [v,f] = readOFF(filename)
    fid = fopen(filename);
    
    title = fscanf(fid,'%s',[1,1]);
    
    nums = fscanf(fid, '%d %d %d',[1,3]);
    
    vn = nums(1);
    fn = nums(2);
    
    v = fscanf(fid, '%f %f %f',[3,vn]);
    f = fscanf(fid, '%d %d %d %d',[4,fn]);
    f = f(2:4,:);
    
    fclose(fid);
end