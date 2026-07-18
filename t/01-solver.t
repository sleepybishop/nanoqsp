use strict;
use warnings;
use Test::More;

my @strategies = (0, 1, 2, 3, 4);
my @strategy_names = ("Coordinate Descent", "Projected Gradient", "Accelerated Gradient", "Spectral Projected Gradient", "ADMM");

plan tests => 30;

for my $s_idx (0..$#strategies) {
    my $strat = $strategies[$s_idx];
    my $s_name = $strategy_names[$s_idx];

    {
        my $output = `./test_runner 1 $strat`;
        chomp($output);
        if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret, $x0, $x1) = ($1, $2, $3);
            ok($ret >= 0, "$s_name - Test 1 status ok");
            my $err = abs($x0 - 0.5) + abs($x1 - 0.5);
            ok($err < 1e-4, "$s_name - Test 1 solution close to [0.5, 0.5]")
                or diag("Got: x0=$x0, x1=$x1 (err=$err)");
        } else {
            fail("$s_name - Test 1 output format mismatch");
        }
    }

    {
        my $output = `./test_runner 2 $strat`;
        chomp($output);
        if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret, $x0, $x1) = ($1, $2, $3);
            ok($ret >= 0, "$s_name - Test 2 status ok");
            my $err = abs($x0 - 1.0) + abs($x1 - 0.0);
            ok($err < 1e-4, "$s_name - Test 2 solution close to [1.0, 0.0]")
                or diag("Got: x0=$x0, x1=$x1 (err=$err)");
        } else {
            fail("$s_name - Test 2 output format mismatch");
        }
    }

    {
        my $output = `./test_runner 3 $strat`;
        chomp($output);
        ok($output =~ /RET:-1/, "$s_name - Test 3 returns error for invalid arguments");
    }

    {
        my $output = `./test_runner 4 $strat`;
        chomp($output);
        if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret, $x0, $x1) = ($1, $2, $3);
            ok($ret >= 0, "$s_name - Test 4 handles singular matrix with regularization");
        } else {
            fail("$s_name - Test 4 output format mismatch");
        }
    }
}
