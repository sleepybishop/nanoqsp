use strict;
use warnings;
use Test::More tests => 14;
use IPC::Open2;

sub run_case {
    my ($id) = @_;
    return `./test_runner $id 0`;
}

my $dense_spg = run_case(17);
like($dense_spg, qr/RET:\d+/, "dense SPG accepts projected warm start");
like($dense_spg, qr/X0:0\.000000/, "dense SPG output is feasible");

my $sparse_spg = run_case(18);
like($sparse_spg, qr/RET:\d+/, "sparse SPG accepts projected warm start");
like($sparse_spg, qr/X0:0\.000000/, "sparse SPG output is feasible");

like(run_case(19), qr/RET:-4/, "iteration exhaustion has a distinct status");
like(run_case(20), qr/RET:-1/, "malformed CSR is rejected");
like(run_case(21), qr/RET:-1/, "negative affine dimension is rejected");

my $affine = run_case(22);
like($affine, qr/RET:\d+/, "dense affine solve converges");
like($affine, qr/X0:1\.000000/, "dense affine output is projected");

like(run_case(23), qr/RET:-1/, "inconsistent bounds are rejected");
like(run_case(24), qr/RET:-2/, "unsupported affine strategy is rejected");
like(run_case(25), qr/RET:0/, "sparse coordinate diagonal remains intact");
like(run_case(26), qr/RET:0/, "workspace size queries report expected sizes");

my $pid = open2(my $out, my $in, "./solve_cli 0");
print {$in} "-1 2\n";
close($in);
<$out>;
waitpid($pid, 0);
is($? >> 8, 1, "CLI rejects negative dimensions");
