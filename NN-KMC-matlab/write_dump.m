function write_dump(i_fname, i_frame, i_step, varargin)
FileMode = 'w';
if size(varargin, 2) >= 1
    if strcmp(varargin{1}, 'append') == 1
        FileMode = 'a';
    end
end

assert(ischar(i_fname), 'the 1st parameter must be string');
assert(isstruct(i_frame), 'the second parameter must be struct');
assert(isfield(i_frame, 'nAtom'), '"nAtom" must be a field of i_frame');

Natom = i_frame.nAtom;
bound = i_frame.bounds;
Data(1, 1:Natom) = 1:Natom;
Data(2, 1:Natom) = i_frame.type;
Data(3, 1:Natom) = i_frame.x;
Data(4, 1:Natom) = i_frame.y;
Data(5, 1:Natom) = i_frame.z;

Data(6, 1:Natom) = i_frame.freq;

Fid = fopen(i_fname, FileMode);
fprintf(Fid, 'ITEM: TIMESTEP\n%d\n', i_step);
fprintf(Fid, 'ITEM: NUMBER OF ATOMS\n%d\n', Natom);
fprintf(Fid, 'ITEM: BOX BOUNDS\n%g %g\n%g %g\n%g %g\n', ...
             bound(1, 1), bound(2, 1), ...
             bound(1, 2), bound(2, 2), ...
             bound(1, 3), bound(2, 3));
fprintf(Fid, 'ITEM: ATOMS id type x y z freq\n');
fprintf(Fid, '%d %d %g %g %g %g\n', Data);

fclose(Fid);

