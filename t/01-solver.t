use strict;
use warnings;
use Test::More;

my @strategies = (0, 1, 2, 3, 4);
my @strategy_names = (
    "Coordinate Descent",
    "Projected Gradient",
    "Accelerated Gradient",
    "Spectral Projected Gradient",
    "ADMM"
);

plan tests => 43;

for my $s_idx (0 .. $#strategies) {
    my $strat = $strategies[$s_idx];
    my $s_name = $strategy_names[$s_idx];

    {
        my $output = `./t/00util/test_runner 1 $strat`;
        chomp($output);
        if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret, $x0, $x1) = ($1, $2, $3);
            ok($ret >= 0, "$s_name - Test 1 status ok");
            my $err = abs($x0 - 0.5) + abs($x1 - 0.5);
            ok($err < 1e-4, "$s_name - Test 1 solution close to [0.5, 0.5]")
                or diag("Got: x0=$x0, x1=$x1 (err=$err)");
        } else {
            fail("$s_name - Test 1 output format mismatch");
            fail("$s_name - Test 1 solution check skipped");
        }
    }

    {
        my $output = `./t/00util/test_runner 2 $strat`;
        chomp($output);
        if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret, $x0, $x1) = ($1, $2, $3);
            ok($ret >= 0, "$s_name - Test 2 status ok");
            my $err = abs($x0 - 1.0) + abs($x1 - 0.0);
            ok($err < 1e-4, "$s_name - Test 2 solution close to [1.0, 0.0]")
                or diag("Got: x0=$x0, x1=$x1 (err=$err)");
        } else {
            fail("$s_name - Test 2 output format mismatch");
            fail("$s_name - Test 2 solution check skipped");
        }
    }

    {
        my $output = `./t/00util/test_runner 3 $strat`;
        chomp($output);
        ok($output =~ /RET:-1/, "$s_name - Test 3 returns error for invalid arguments");
    }

    {
        my $output = `./t/00util/test_runner 4 $strat`;
        chomp($output);
        if ($output =~ /RET:(-?\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
            my ($ret) = ($1);
            ok($ret >= 0 || $ret == -4,
                "$s_name - Test 4 handles singular matrix without false success");
        } else {
            fail("$s_name - Test 4 output format mismatch");
        }
    }

    {
        my $output = `./t/00util/test_runner 10 $strat`;
        chomp($output);
        ok($output =~ /RET:\d+/, "$s_name - Test 10 bounded least squares converges");
    }
}

for my $case (
    [5, "Test 5 (N=10 Randomized Bounded QP)"],
    [6, "Test 6 (N=50 Randomized Bounded QP)"],
    [7, "Test 7 (N=10 Unconstrained QP)"],
    [8, "Test 8 (N=10 Semi-constrained QP)"],
    [9, "Test 9 (N=10 Ill-conditioned QP)"],
) {
    my ($id, $desc) = @$case;
    my $output = `./t/00util/test_runner $id`;
    chomp($output);
    ok($output =~ /RET:0/, "Cross-Strategy Verification - $desc");
}

{
    my $output = `./t/00util/test_runner 15 4`;
    chomp($output);
    if ($output =~ /RET:(\d+)\s+X0:([\d\.-]+)\s+X1:([\d\.-]+)/) {
        my ($ret, $x0, $x1) = ($1, $2, $3);
        ok($ret >= 0, "ADMM - Test 15 (Dense General Affine QP) status ok");
        my $err = abs($x0 - 0.142857) + abs($x1 - 0.678571);
        ok($err < 1e-4, "ADMM - Test 15 solution close to [0.142857, 0.678571]")
            or diag("Got: x0=$x0, x1=$x1 (err=$err)");
    } else {
        fail("ADMM - Test 15 output format mismatch");
        fail("ADMM - Test 15 solution check skipped");
    }
}

{
    my $ex_out = `./examples/least_squares`;
    is($? >> 8, 0, "examples/least_squares runs cleanly") or diag($ex_out);
}
