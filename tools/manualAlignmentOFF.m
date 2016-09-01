function manualAlignmentCT(rotationMatrix,file1,file2)
    [v_raw,f_raw] = readOFF(file1);
	A = load(rotationMatrix); 
    V = A*[v_raw;ones(1,size(v_raw,2))];
    writeOFF(V(1:3,:),f_raw,file2);
end