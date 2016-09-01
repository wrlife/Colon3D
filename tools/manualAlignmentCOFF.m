function manualAlignmentCOFF(rotationMatrix,file1,file2)
    [v_raw,c_raw,f_raw] = readCOFF(file1);
	A = load(rotationMatrix); 
    V = A*[v_raw;ones(1,size(v_raw,2))];
    writeCOFF(V(1:3,:),c_raw,f_raw,file2);
end