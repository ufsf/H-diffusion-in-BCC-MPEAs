function [Hx_unwarp, Hy_unwarp, Hz_unwarp] = unwrap_Coord(H_diff, Box)
    Hx_unwarp = processComponent(H_diff(:,1), Box);
    Hy_unwarp = processComponent(H_diff(:,2), Box);
    Hz_unwarp = processComponent(H_diff(:,3), Box);

    function Hu_unwarp = processComponent(Hu, Box)
        dHu = diff(Hu);
        dHu_old = dHu;

        dHu(dHu_old > Box/2) = dHu(dHu_old > Box/2) - Box;
        dHu(dHu_old < -Box/2) = dHu(dHu_old < - Box/2) + Box;

        Hu_unwarp = zeros(size(Hu));
        Hu_unwarp(1) = Hu(1);

        Hu_unwarp(2:end) = Hu_unwarp(1) + cumsum(dHu);
    end
end
