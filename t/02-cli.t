use strict;
use warnings;
use Test::More tests => 25;
use IPC::Open2;

my $input = <<'EOF';
4 2
0.5 5.0
1.0 0.0 1.1
1.0 0.0 0.9
2.0 1.0 3.2
0.0 1.0 1.3
1.0 1.0
EOF

my @strategies = (0, 1, 2, 3, 4);
my @strategy_names = ("Coordinate Descent", "Projected Gradient", "Accelerated Gradient", "Spectral Projected Gradient", "ADMM");

for my $s_idx (0..$#strategies) {
    my $strat = $strategies[$s_idx];
    my $s_name = $strategy_names[$s_idx];

    my $pid = open2(my $chld_out, my $chld_in, "./solve_cli $strat");
    print $chld_in $input;
    close($chld_in);

    my @output = <$chld_out>;
    waitpid($pid, 0);
    my $exit_code = $? >> 8;

    ok($exit_code == 0, "$s_name - solve_cli ran successfully");

    my $output_str = join("", @output);

    ok($output_str =~ /Iterations: \d+/, "$s_name - Output contains iterations count");

    if ($output_str =~ /x\[0\] = ([\d\.]+)/) {
        my $x0 = $1;
        ok(abs($x0 - 0.975) < 0.1, "$s_name - Apple price solved to approx 0.975");
    } else {
        fail("$s_name - Could not find x[0] in output");
    }

    if ($output_str =~ /x\[1\] = ([\d\.]+)/) {
        my $x1 = $1;
        ok(abs($x1 - 1.275) < 0.1, "$s_name - Banana price solved to approx 1.275") or diag("Output: $output_str");
    } else {
        fail("$s_name - Could not find x[1] in output");
    }

    if ($output_str =~ /Prediction = ([\d\.]+)/) {
        my $pred = $1;
        ok(abs($pred - 2.25) < 0.1, "$s_name - Prediction for 1 Apple + 1 Banana is approx 2.25");
    } else {
        fail("$s_name - Could not find Prediction in output");
    }
}
