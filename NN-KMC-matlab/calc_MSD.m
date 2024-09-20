function [my_MSD,H_unwarp,coord_final,Time_extract_end,Time_extract_fake_end] =...
    calc_MSD(H_diff,Box, Len,Time, Time_fake)

[Hx_unwarp, Hy_unwarp, Hz_unwarp] = unwrap_Coord(H_diff, Box);

H_unwarp(:,1) = Hx_unwarp;
H_unwarp(:,2) = Hy_unwarp;
H_unwarp(:,3) = Hz_unwarp;

if length(Hx_unwarp)~=length(Time)
    error('Wrong MSD!!');
end

rng('shuffle');
start_point = randi([fix(length(Hx_unwarp)/10),fix(length(Hx_unwarp)*9/10)]);

Hx_unwarp_1 = Hx_unwarp(1:start_point-1);
Hy_unwarp_1 = Hy_unwarp(1:start_point-1);
Hz_unwarp_1 = Hz_unwarp(1:start_point-1);
Hx_unwarp_2 = Hx_unwarp(start_point:end);
Hy_unwarp_2 = Hy_unwarp(start_point:end);
Hz_unwarp_2 = Hz_unwarp(start_point:end);

%%
Time1 = Time(1:start_point-1);
Time1_fake = Time_fake(1:start_point-1,:);
Hx_cur = Hx_unwarp_1(1);
Hy_cur = Hy_unwarp_1(1);
Hz_cur = Hz_unwarp_1(1);
count = 1;

Hx_extract1 = [];
Hy_extract1 = [];
Hz_extract1 = [];
Time_extract1 = [];
Time_extract1_fake = [];
for i = 1:length(Hx_unwarp_1)
    Hx_unwarp_tmp  = Hx_unwarp_1(i);
    Hy_unwarp_tmp  = Hy_unwarp_1(i);
    Hz_unwarp_tmp  = Hz_unwarp_1(i);
    dist_tmp1 = Hx_unwarp_tmp - Hx_cur;
    dist_tmp2 = Hy_unwarp_tmp - Hy_cur;
    dist_tmp3 = Hz_unwarp_tmp - Hz_cur;
    dist_tmp = (dist_tmp1.^2+dist_tmp2.^2+dist_tmp3.^2).^0.5;
    if dist_tmp > Len
        Hx_cur = Hx_unwarp_tmp;
        Hy_cur = Hy_unwarp_tmp;
        Hz_cur = Hz_unwarp_tmp;
        Hx_extract1(count) = Hx_unwarp_tmp;
        Hy_extract1(count) = Hy_unwarp_tmp;
        Hz_extract1(count) = Hz_unwarp_tmp;
        Time_extract1(count) = Time1(i);
        Time_extract1_fake(count,:) = Time1_fake(i,:);
        count = count + 1;
    end
end

Time2 = Time(start_point:end);
Time2_fake = Time_fake(start_point:end,:);

Hx_cur = Hx_unwarp_2(1);
Hy_cur = Hy_unwarp_2(1);
Hz_cur = Hz_unwarp_2(1);
count = 1;

Hx_extract2 = [];
Hy_extract2 = [];
Hz_extract2 = [];
Time_extract2 = [];
Time_extract2_fake = [];

for i = 1:length(Hx_unwarp_2)
    Hx_unwarp_tmp  = Hx_unwarp_2(i);
    Hy_unwarp_tmp  = Hy_unwarp_2(i);
    Hz_unwarp_tmp  = Hz_unwarp_2(i);
    dist_tmp1 = Hx_unwarp_tmp - Hx_cur;
    dist_tmp2 = Hy_unwarp_tmp - Hy_cur;
    dist_tmp3 = Hz_unwarp_tmp - Hz_cur;
    dist_tmp = (dist_tmp1.^2+dist_tmp2.^2+dist_tmp3.^2).^0.5;
    if dist_tmp > Len
        Hx_cur = Hx_unwarp_tmp;
        Hy_cur = Hy_unwarp_tmp;
        Hz_cur = Hz_unwarp_tmp;
        Hx_extract2(count) = Hx_unwarp_tmp;
        Hy_extract2(count) = Hy_unwarp_tmp;
        Hz_extract2(count) = Hz_unwarp_tmp;
        Time_extract2(count) = Time2(i);
        Time_extract2_fake(count,:) = Time2_fake(i,:);
        count = count + 1;
    end
end
Hx_extract = [Hx_extract1 Hx_extract2];
Hy_extract = [Hy_extract1 Hy_extract2];
Hz_extract = [Hz_extract1 Hz_extract2];
Time_extract = [Time_extract1 Time_extract2];
Time_extract_fake = [Time_extract1_fake;Time_extract2_fake];

Hx_extract_diff = diff(Hx_extract);
Hy_extract_diff = diff(Hy_extract);
Hz_extract_diff = diff(Hz_extract);

my_MSD = cumsum(Hx_extract_diff.^2 + Hy_extract_diff.^2 + Hz_extract_diff.^2);
coord_final.x = Hx_extract;
coord_final.y = Hy_extract;
coord_final.z = Hz_extract;
if isempty(Time_extract)
    Time_extract_end = nan;
    Time_extract_fake_end = nan;
else
    Time_extract_end = Time_extract(end);
    Time_extract_fake_end = Time_extract_fake(end,:);
end
end