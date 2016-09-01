function [I,origin,spacing] = loadMETA(filename, verbose)
%
% loads 2 and 3D META images and vector fields
%
if nargin < 2
  verbose = false;
end
verbose = false;
dataFilename='';
dataType = '';
imSize = [0 0 0];
numChannels = 1;
origin = [0 0 0];
spacing = [1 1 1];
headerSize= 0;
% parse header file
%fprintf('Loading image: %s...\n',filename);
[key,val] = textread(filename,'%s=%[^\n]');
for i=1:size(key,1)
  switch key{i}
   case 'ObjectType'
     if verbose
       fprintf('  Object Type: %s\n', val{i});
     end
   case 'NDims'
     if verbose     
       fprintf('  Number of Dimensions: %s\n', val{i});
     end
   case 'DimSize'
     if verbose
       fprintf('  Size: %s\n', val{i});
     end
     imSize = str2num(val{i});
   case 'ElementType'
     if verbose
       fprintf('  Element Type: %s\n', val{i});
     end
     dataType = decideMETADataType(val{i});
   case 'ElementDataFile'
     if verbose
       fprintf('  DataFile: %s\n', val{i});
     end
    dataFilename = val{i};
   case 'ElementNumberOfChannels'
     if verbose
       fprintf('  Number of Channels: %s\n', val{i});
     end
    numChannels = str2num(val{i});
   case {'Offset','Position'}
     if verbose
       fprintf('  Offset: %s\n', val{i});
     end
    origin = str2num(val{i});
   case 'ElementSpacing'
     if verbose
       fprintf('  Spacing: %s\n', val{i});
     end
    spacing = str2num(val{i});
   case 'HeaderSize'
      if verbose
       fprintf('  HeaderSize: %s\n', val{i});
      end
     headerSize = str2num(val{i});
      otherwise
     if verbose
       fprintf('  Unknown Key/Val: %s/%s\n',key{i},val{i});
     end
  end
end

% load image data
path = fileparts(filename);
fid = fopen([path filesep dataFilename],'r');
if (fid==-1)
  error(sprintf('Can''t read file: %s\n', [path filesep dataFilename]));
end


  
if (headerSize ~= 0)
    fseek(fid, headerSize, 'bof');
else
    % header size is not specified, read from the end of the file
    % when raw3--> meta, the headersize is not specified, (should be 88 for lung images)
    switch dataType
        case {'float32','float', 'uint32'}
            bytes = 4;
        case {'uint8'}
            bytes=1;
        case{'uint16','int16'}
            bytes =2;
        case{'float64'}
            bytes = 8;
    end

    bytesExpected = prod(imSize)*numChannels*bytes;
    fseek( fid, -bytesExpected, 'eof' );

end

headerSize = ftell(fid);

[I,count] = fread(fid,prod(imSize)*numChannels,dataType);
fclose(fid);
I=squeeze(reshape(I,[numChannels imSize]));

