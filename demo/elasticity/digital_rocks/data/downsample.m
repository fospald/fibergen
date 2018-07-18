
function downsample()

% Data from: https://github.com/fkrzikalla/drp-benchmarks

data = raw2matlab('berea_vsg_1024x1024x1024.raw', [1024,1024,1024], 'uint8');
data = data(151:874, 151:874, :);
matlab2raw(data, 'berea_vsg_724x724x1024.raw', 'uint8');

downsample_file_default('berea_vsg_724x724x1024.raw');

return

downsample_file_default('fontainebleau_exxon_288x288x300.raw');
downsample_file_default('berea_kongju_1024x1024x1024.raw');
downsample_file_default('berea_vsg_1024x1024x1024.raw');
downsample_file_default('berea_stanford_724x724x1024.raw');
downsample_file_default('grosmont_kongju_1024x1024x1024.raw');
downsample_file_default('grosmont_vsg_1024x1024x1024.raw');
downsample_file_default('grosmont_stanford_1024x1024x1024.raw');
downsample_file_default('spherepack_788x791x793.raw');

return


function downsample_file_default(filename)

fprintf('downsampling %s\n', filename);
c = strsplit(filename, '_');
s = strsplit(c{length(c)}, 'x');
w = strsplit(s{length(s)}, '.');
shape = [str2double(s{1}),str2double(s{2}),str2double(w{1})];
fmt = sprintf('%s_%%dx%%dx%%d_%%d.raw', strjoin( c(1:(length(c)-1)), '_'));
downsample_file(filename, shape, fmt);

return


function downsample_file(filename, shape, fmt)

data_org = raw2matlab(filename, shape, 'uint8');
pmin = min(min(min(data_org)));
pmax = max(max(max(data_org)));

for p=pmin:pmax
    data = (data_org == p);
    n = size(data);
    while 1
        filename = sprintf(fmt, n(1), n(2), n(3), p);
        if ~exist(filename, 'file')
            matlab2raw(data*255, filename, 'uint8');
        end
        if (any(mod(n, 2)))
            break;
        end
        n = n/2;
        data_new = zeros(n);
        i1 = 1:n(1);
        i2 = 1:n(2);
        i3 = 1:n(3);
        data_new(i1, i2, i3) = (1/8.0)*( ...
            data(i1*2-0, i2*2-0, i3*2-0) + data(i1*2-0, i2*2-0, i3*2-1) + ...
            data(i1*2-0, i2*2-1, i3*2-0) + data(i1*2-0, i2*2-1, i3*2-1) + ...
            data(i1*2-1, i2*2-0, i3*2-0) + data(i1*2-1, i2*2-0, i3*2-1) + ...
            data(i1*2-1, i2*2-1, i3*2-0) + data(i1*2-1, i2*2-1, i3*2-1) );
        data = data_new;
    end
end

