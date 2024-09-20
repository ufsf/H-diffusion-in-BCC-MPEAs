function save_dump(H_wrap,imat,N, fname)
Len = load(['len',num2str(N),'.txt']);
Len_pure = load('len_pure.txt');
if imat > 0
    Box = Len(imat);
else
    Box = Len_pure(abs(imat))*N;
end
% H_wrap = res.H_wrap;
[unique_points, ~, ic] = unique(H_wrap, 'rows', 'stable');
point_counts = accumarray(ic, 1)/length(H_wrap);
id = 1:length(unique_points);
type = ones(size(id));
bounds = [0 0 0; Box Box Box];
frame.nAtom = length(unique_points);
frame.bounds = bounds;
frame.type = type;
frame.x = unique_points(:,1);
frame.y = unique_points(:,2);
frame.z = unique_points(:,3);
frame.freq = point_counts;
write_dump(fname, frame, 1)