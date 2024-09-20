function my_MSD = calc_MSD_one(H_diff,Box, Len,Time)

[Hx_unwarp, Hy_unwarp, Hz_unwarp] = unwrap_Coord(H_diff, Box);

if length(Hx_unwarp)~=length(Time)
    error('Wrong MSD!!');
end

%%
Hx_cur = Hx_unwarp(1);
Hy_cur = Hy_unwarp(1);
Hz_cur = Hz_unwarp(1);
count = 1;

Hx_extract = [];
Hy_extract = [];
Hz_extract = [];

for i = 1:length(Hx_unwarp)
    Hx_unwarp_tmp  = Hx_unwarp(i);
    Hy_unwarp_tmp  = Hy_unwarp(i);
    Hz_unwarp_tmp  = Hz_unwarp(i);
    dist_tmp1 = Hx_unwarp_tmp - Hx_cur;
    dist_tmp2 = Hy_unwarp_tmp - Hy_cur;
    dist_tmp3 = Hz_unwarp_tmp - Hz_cur;
    dist_tmp = (dist_tmp1.^2+dist_tmp2.^2+dist_tmp3.^2).^0.5;
    if dist_tmp > Len
        Hx_cur = Hx_unwarp_tmp;
        Hy_cur = Hy_unwarp_tmp;
        Hz_cur = Hz_unwarp_tmp;
        Hx_extract(count) = Hx_unwarp_tmp;
        Hy_extract(count) = Hy_unwarp_tmp;
        Hz_extract(count) = Hz_unwarp_tmp;
        count = count + 1;
    end
end

Hx_extract_diff = diff(Hx_extract);
Hy_extract_diff = diff(Hy_extract);
Hz_extract_diff = diff(Hz_extract);

my_MSD = cumsum(Hx_extract_diff.^2 + Hy_extract_diff.^2 + Hz_extract_diff.^2);
end