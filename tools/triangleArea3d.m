function area = triangleArea3d(pt1, pt2, pt3)

% compute individual vectors
v12 = pt2 - pt1;
v13 = pt3 - pt1;

% compute area from cross product
area = norm(cross(v12, v13), 2) / 2;
end