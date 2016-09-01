function color = interpolate2D(x,y,im)
    xf = floor(x) - (abs(x-240)<0.000001); xc = xf+1;
    yf = floor(y) - (abs(y-720)<0.000001); yc = yf+1;
    
    xweight = x - xf; yweight = y - yf;
    
    c1f = im(xf,yf,1) * (1-yweight) + im(xf,yc,1) * yweight;
    c1c = im(xc,yf,1) * (1-yweight) + im(xc,yc,1) * yweight;
    c1 = c1f * (1-xweight) + c1c * xweight;
    
    c2f = im(xf,yf,2) * (1-yweight) + im(xf,yc,2) * yweight;
    c2c = im(xc,yf,2) * (1-yweight) + im(xc,yc,2) * yweight;
    c2 = c2f * (1-xweight) + c2c * xweight;
    
    c3f = im(xf,yf,3) * (1-yweight) + im(xf,yc,3) * yweight;
    c3c = im(xc,yf,3) * (1-yweight) + im(xc,yc,3) * yweight;
    c3 = c3f * (1-xweight) + c3c * xweight;
    
    color = round([c1;c2;c3]);
end