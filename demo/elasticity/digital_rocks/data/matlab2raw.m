
function matlab2raw(data3d, filename, precision)
fprintf('writing %s\n', filename);
fh = fopen(filename, 'wb+');
fwrite(fh, data3d, precision);
fclose(fh);
return;
