
function data3d = raw2matlab(filename, shape, precision)
fh = fopen(filename, 'rb');
fmt = sprintf('%s=>%s', precision, precision);
data3d = fread(fh, prod(shape), fmt); 
data3d = reshape(data3d, shape);
fclose(fh);
return;
