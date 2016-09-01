function [X,Y,Z] = convertBin2Surf(filename,color_filename,Q,Translation,output_filename,outputFlag)
    %% parameters
    %fx = 638.247; fy = 269.796; cx = 376.04; cy = 100.961;
    fx = 726.6843; fy = 363.2936; cx = 657.2408; cy =  269.2426;

    w = 1103; h = 381;
    %scale = 2.25;
    scale = 1;
    %% read & convert point positions
    fid = fopen(filename,'r');
    b = fread(fid,w*h,'float');
    fclose(fid);
    z = vec2mat(b,w);
    x = repmat([0:w-1],h,1);
    y = repmat([0:h-1]',1,w);
    x = (x - cx) / fx;
    y = (y - cy) / fy;
    X = x .* z * scale;
    Y = y .* z * scale;
    Z = z * scale;
    
    X = X - Translation(1);
    Y = Y - Translation(2);
    Z = Z - Translation(3);
    
    V = [X(:)';Y(:)';Z(:)'];
    V = RotateVector(quaternion(Q).inverse,V);
    X = vec2mat(V(1,:),h)';
    Y = vec2mat(V(2,:),h)';
    Z = vec2mat(V(3,:),h)';
    
    %% create COFF files
    if (outputFlag)
    
       %% read colors
        c = imread(color_filename);
        
        fid = fopen(output_filename,'w');
        fprintf(fid,'COFF\n');
        fprintf(fid,'%d %d 0\n',w*h,(w-1)*(h-1)*2);
    
        for i = 1:h
            for j = 1:w
                fprintf(fid,'%f %f %f %d %d %d 255\n', X(i,j),Y(i,j),Z(i,j),c(i,j,1),c(i,j,2),c(i,j,3));
            end
        end
    
        for i = 1:h-1
            for j = 1:w-1
                fprintf(fid,'3 %d %d %d\n', (i-1)*w+j-1, (i-1)*w+j, i*w+j-1);
                fprintf(fid,'3 %d %d %d\n', i*w+j-1, (i-1)*w+j, i*w+j);
            end
        end
        fclose(fid);
    end
    
end