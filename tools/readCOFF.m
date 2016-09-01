function [v,c,f] = readCOFF(filename)
    fid = fopen(filename);
    
    title = fscanf(fid,'%s',[1,1]);
    
    nums = fscanf(fid, '%d %d %d',[1,3]);
    
    vn = nums(1);
    fn = nums(2);
    
    vc = fscanf(fid, '%f %f %f %d %d %d %d',[7,vn]);
    v = vc(1:3,:);
    c = vc(4:6,:);
    
    f = fscanf(fid, '%d %d %d %d',[4,fn]);
    f = f(2:4,:);
    
    fclose(fid);
end