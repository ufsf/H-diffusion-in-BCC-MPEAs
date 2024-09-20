function final_D = CalcD(imat,Temp,MSD_max, Size)
%%
% input:
% imat - material id
% Temp - simulation temperature
% MSD_max - maximum MSD
% Size - number of sampling oringinal trajectory 
%% Read box
Box_tmp = 3.2*Size;
Len = load(['len',num2str(Size),'.txt']);
Len_pure = load('len_pure.txt');
if imat > 0
    Box = Len(imat);
else
    Box = Len_pure(abs(imat))*Size;
end
%% Read diffusion path, and Box size and H coordinates
All_path_id = load(['../H-Path/path',num2str(Size),'/All_TT.txt']);
All_path_id_1 = All_path_id(:,1);
All_path_id_2 = All_path_id(:,2);
num_site = length(unique(All_path_id_1));
Ratio = Box/Box_tmp;
H_coords = load(['../H-Path/path',num2str(Size),'/H_coords.txt']);
H_coords = H_coords*Ratio;
%% read barriers and SE
% npy is the data numpy 
Barrier = readNPY(['../All-Barrier-',num2str(Size),'/',num2str(abs(imat)),'.npy']);
Barrier = exp(Barrier);
SE = readNPY(['../All-Barrier-',num2str(Size),'/',num2str(abs(imat)),'-TSE.npy']);
if imat == -1 % Mo
    Barrier = 0.1576*ones(size(Barrier));
    SE = 0*ones(size(SE));
elseif imat== -2 % Nb
    Barrier = 0.1623*ones(size(Barrier));
    SE = 0*ones(size(SE));
elseif imat == -3 % Ta
    Barrier = 0.1934*ones(size(Barrier));
    SE = 0*ones(size(SE));
elseif imat == -4 % W
    Barrier = 0.1933*ones(size(Barrier));
    SE = 0*ones(size(SE));
end
%% KMC parameters
kb = 8.617333262e-5;
v0 = 1.5e13;
Critical_len_MSD = 20;
Temp_fake_all = 400:100:5000;
%% initilize states
rng('shuffle');
curr_id = randperm(num_site,1);
Time = 0;
Time_fake_all = zeros(size(Temp_fake_all));
H_wrap = [];
H_wrap(1,:) = H_coords(curr_id,:);
Time_store = [];
Time_store_fake = [];
Time_store(1) = 0;
SE_store = [];
DB_store = [];
count = 1;
%% run KMC
while 1
    SE_store = [SE_store SE(curr_id)];
    index_temp = find(All_path_id_1==curr_id);
    next = All_path_id_2(index_temp);
    barrier_temp = Barrier(index_temp);
    All_rate = v0*exp(-barrier_temp/(kb*Temp));
    All_rate_fake = v0*exp(-barrier_temp./(kb*Temp_fake_all));
    SumRate = sum(All_rate);
    SumRate_fake = sum(All_rate_fake);
    Cumulative_sum = cumsum(All_rate);
    randRate = rand()*SumRate;
    index = find(Cumulative_sum>=randRate);
    index = index(1); % Only this event will occur in All_rate
    DB_store = [DB_store barrier_temp(index)];
    curr_id = next(index);
    temp_rand = rand();
    Time = Time - log(temp_rand)/SumRate;
    Time_fake_all = Time_fake_all - log(temp_rand)./SumRate_fake;
    Time_store(count+1) = Time;
    Time_store_fake(count+1,:) = Time_fake_all;
    H_wrap(count+1,:) = H_coords(curr_id,:);
    count = count + 1;
    %%
    if mod(count,1000) == 0
        MSD_tmp = calc_MSD_one(H_wrap,Box,Critical_len_MSD,Time_store);
        if ~isempty(MSD_tmp)
            disp(imat);
            fprintf('Current temperature %d K --- progess %.8f\n',...
                Temp, MSD_tmp(end)/MSD_max)
            if MSD_tmp(end)/MSD_max > 100
                error('Wrong!!!');
            end
            if MSD_tmp(end) > MSD_max
                break;
            end
        end
    end
end
%%
final_MSD = calc_MSD_one(H_wrap,Box,Critical_len_MSD,Time_store);
final_D = final_MSD(end)/6/Time_store(end)*1e-20;
end