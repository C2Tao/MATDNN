function dy = DIFF(y)

% first order derivative of y

y = [y(1), y(1), y, y(end), y(end)];

dy = (2*y(5:end) + y(4:end-1) - y(2:end-3) - 2*y(1:end-4))/10;
