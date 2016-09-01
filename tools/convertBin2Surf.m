function cut_file = convertBin2Surf(filename,color_filename,Q,Translation,CameraParam,output_filename,outputFlag)
    %% parameters
    %fx = 638.247; fy = 269.796; cx = 376.04; cy = 100.961;
    %fx =444.569; fy = 194.111; cx = 338.0543; cy =  134.8415;
    
    %fx=324.306; fy=154.45; cx=338.054; cy=134.842;
    
    %fx=365.57; fy=163.115;cx= 338.054 ;cy=134.842;
    fx=CameraParam.fx; fy=CameraParam.fy;cx=CameraParam.cx; cy=CameraParam.cy;
    %fx=567.239; fy=261.757;cx= 376.04;cy= 100.961;
   
    
    depthThreshold = 500;
    colorTreshold = 20;
    
    
    %%
    w = CameraParam.w; h = CameraParam.h;
    %w=720;h=240;
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
    
    cut_point = detectCutRegion(X,Y,Z,0.18);
    cut_point = cut_point | (Z > depthThreshold);
    %depthcut=cut_point;
    depthcut=zeros(size(cut_point));
    
    X = X - Translation(1);
    Y = Y - Translation(2);
    Z = Z - Translation(3);
    
    V = [X(:)';Y(:)';Z(:)'];
    V = RotateVector(quaternion(Q).inverse,V);
    X = vec2mat(V(1,:),h)';
    Y = vec2mat(V(2,:),h)';
    Z = vec2mat(V(3,:),h)';
    
    %cut_point = cut_point | (Z > depthThreshold);
    %% create COFF files
    if (outputFlag)
        cut_face = 0;
       %% read colors
        c = imread(color_filename);
        
        gimg=rgb2gray(c);
         cut_point = cut_point | (gimg < colorTreshold);
        
         rcount = 0;
        for i = 1:h-1
            for j = 1:w-1
                if (depthcut(i,j) == 1)|| (depthcut(i+1,j) == 1)|| (depthcut(i,j+1) == 1) || (depthcut(i+1,j+1) == 1)
                    rcount = rcount + 1;
                end
            end
        end
        fid = fopen(output_filename,'w');
        fprintf(fid,'COFF\n');
        fprintf(fid,'%d %d 0\n',w*h,(w-1)*(h-1)*2-rcount*2);
        %fprintf(fid,'%d %d 0\n',w*h,);
        for i = 1:h
            for j = 1:w
                    fprintf(fid,'%f %f %f %d %d %d 255\n', X(i,j),Y(i,j),Z(i,j),c(i,j,1),c(i,j,2),c(i,j,3));
                    %fprintf(fid,'%f %f %f\n', X(i,j),Y(i,j),Z(i,j));
            end
        end
    
        for i = 1:h-1
            for j = 1:w-1
                if (cut_point(i,j) == 1)|| (cut_point(i+1,j) == 1)|| (cut_point(i,j+1) == 1) || (cut_point(i+1,j+1) == 1)
                    cut_face = cut_face + 1;
                     if (depthcut(i,j) == 1)|| (depthcut(i+1,j) == 1)|| (depthcut(i,j+1) == 1) || (depthcut(i+1,j+1) == 1)
                        continue;
                     end
                end
 
                fprintf(fid,'3 %d %d %d\n', (i-1)*w+j-1, (i-1)*w+j, i*w+j-1);
                fprintf(fid,'3 %d %d %d\n', i*w+j-1, (i-1)*w+j, i*w+j);
            end
        end
        fclose(fid);
        
        %%
        cut_file = strcat(output_filename,'.cut.off');
        fid = fopen(cut_file,'w');
        fprintf(fid,'COFF\n');
        fprintf(fid,'%d %d 0\n',w*h,(w-1)*(h-1)*2 - cut_face*2);
        %fprintf(fid,'%d %d 0\n',1,0);
        %fprintf(fid,'%f %f %f %d %d %d 255\n', X(1,1),Y(1,1),Z(1,1),c(1,1,1),c(1,1,2),c(1,1,3));
        for i = 1:h
            for j = 1:w
                    fprintf(fid,'%f %f %f %d %d %d 255\n', X(i,j),Y(i,j),Z(i,j),c(i,j,1),c(i,j,2),c(i,j,3));
            end
        end
    
        for i = 1:h-1
            for j = 1:w-1
                if (cut_point(i,j) == 1)|| (cut_point(i+1,j) == 1)|| (cut_point(i,j+1) == 1) || (cut_point(i+1,j+1) == 1)
                else
                    fprintf(fid,'3 %d %d %d\n', (i-1)*w+j-1, (i-1)*w+j, i*w+j-1);
                    fprintf(fid,'3 %d %d %d\n', i*w+j-1, (i-1)*w+j, i*w+j);
                end
            end
        end
        fclose(fid);
    end
    
end