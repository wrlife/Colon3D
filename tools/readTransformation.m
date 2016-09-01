function [T,Name] = readTransformation(filename)
    fid = fopen(filename,'r');
    
    tline = fgets(fid);
    
    i = 1;
    while ischar(tline)
        pos = findstr(tline, 'frame');
        if (length(pos) > 0)
            %Name(i) = str2num(tline(length(tline)-9:length(tline)-6));
            Name(i) = str2num(tline(length(tline)-8:length(tline)-5));
            t = sscanf(tline,'%d %f %f %f %f %f %f %f %d');
            T(i,:) = t;
            i = i + 1;
        end
        tline = fgets(fid);
    end
    fclose(fid);
end