use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Spec;
use IPC::Open3;
use Symbol qw(gensym);
use POSIX qw(pathconf _PC_PATH_MAX);

my $TRAIN   = "./examples/svm/svm_train";
my $PREDICT = "./examples/svm/svm_predict";
my $SIMPLE  = "1 1:2\n1 1:1\n-1 1:-1\n-1 1:-2\n";
my $HEADER  = "solver 0\nkernel 0\ngamma 0\nbias 0\nlabels 1 -1\nnr_sv 2\nmax_dim 1\n";

sub write_file {
    my ($path, $contents) = @_;
    open my $fh, '>', $path or die "cannot write $path: $!";
    print {$fh} $contents;
    close $fh or die "cannot close $path: $!";
    return $path;
}

sub read_file {
    my ($path) = @_;
    open my $fh, '<', $path or die "cannot read $path: $!";
    local $/;
    my $contents = <$fh>;
    close $fh;
    return $contents;
}

sub run_cmd {
    my ($args, $expect_success, $desc) = @_;
    my $err = gensym;
    my $pid = open3(my $in, my $out, $err, @$args);
    close $in;
    local $/;
    my $stdout = <$out> // '';
    my $stderr = <$err> // '';
    waitpid($pid, 0);
    my $code = $? >> 8;
    my $sig  = $? & 127;
    my $diag = "cmd: @$args\nstdout: $stdout\nstderr: $stderr";

    if ($expect_success) {
        ok($code == 0 && $sig == 0, $desc) or diag($diag);
    } else {
        ok($code > 0 && $sig == 0 && $stderr !~ /AddressSanitizer|runtime error:/, $desc)
            or diag($diag);
    }
    return ($stdout, $stderr, $code);
}

sub parse_predictions {
    my ($path) = @_;
    my $text = read_file($path);
    my @rows;
    for my $line (split /\n/, $text) {
        next unless $line =~ /\S/;
        my ($label, $val) = split /\s+/, $line;
        push @rows, [$label + 0, $val + 0];
    }
    return @rows;
}

sub is_finite {
    my ($x) = @_;
    return defined($x) && $x !~ /nan|inf/i && ($x - $x == 0);
}

sub check_labels {
    my ($preds, $expected, $desc) = @_;
    my @got = map { $_->[0] } @$preds;
    my $all_finite = 1;
    for my $p (@$preds) {
        $all_finite = 0 unless is_finite($p->[1]);
    }
    is_deeply(\@got, $expected, "$desc labels");
    ok($all_finite, "$desc finite decision values");
}

my $root   = tempdir("nanoqsp-svm-XXXXXX", TMPDIR => 1, CLEANUP => 1);
my $data   = File::Spec->catfile($root, "data");
my $model  = File::Spec->catfile($root, "model");
my $output = File::Spec->catfile($root, "predictions");

# 1. all solver, kernel, strategy combinations
write_file($data, $SIMPLE);
for my $solver (0, 1) {
    for my $kernel (0, 1) {
        my @strats = $solver == 0 ? (0 .. 4) : (3, 4);
        for my $strat (@strats) {
            my $tag = "s=$solver t=$kernel m=$strat";
            my ($stdout) = run_cmd(
                [$TRAIN, "-s", $solver, "-t", $kernel, "-m", $strat, $data, $model],
                1, "train $tag"
            );
            like($stdout, qr/Training accuracy: 100\.00%/, "accuracy $tag");
            run_cmd([$PREDICT, $data, $model, $output], 1, "predict $tag");
            my @preds = parse_predictions($output);
            check_labels(\@preds, [1, 1, -1, -1], $tag);
        }
    }
}

# 2. small linear coefficients retain their contribution
write_file($data, "1 1:10000\n-1 1:-10000\n");
for my $solver (0, 1) {
    run_cmd([$TRAIN, "-s", $solver, $data, $model], 1, "small linear train s=$solver");
    run_cmd([$PREDICT, $data, $model, $output], 1, "small linear predict s=$solver");
    my @preds = parse_predictions($output);
    check_labels(\@preds, [1, -1], "small linear s=$solver");
    ok(abs($preds[0][1]) > 0.9 && abs($preds[1][1]) > 0.9, "small linear margin s=$solver");
}

# 3. small cost retains rbf support vectors
write_file($data, "1 1:1\n-1 1:-1\n");
for my $solver (0, 1) {
    run_cmd(
        [$TRAIN, "-s", $solver, "-t", 1, "-c", "1e-7", "-e", "1e-12", $data, $model],
        1, "small cost rbf train s=$solver"
    );
    run_cmd([$PREDICT, $data, $model, $output], 1, "small cost rbf predict s=$solver");
    my @preds = parse_predictions($output);
    check_labels(\@preds, [1, -1], "small cost rbf s=$solver");
}

# 4. rbf is invariant to large translation
for my $solver (0, 1) {
    write_file($data, "1 1:0\n-1 1:1\n");
    run_cmd([$TRAIN, "-s", $solver, "-t", 1, "-g", 1, $data, $model], 1, "rbf centered train s=$solver");
    run_cmd([$PREDICT, $data, $model, $output], 1, "rbf centered predict s=$solver");
    my @centered = parse_predictions($output);

    write_file($data, "1 1:100000000\n-1 1:100000001\n");
    run_cmd([$TRAIN, "-s", $solver, "-t", 1, "-g", 1, $data, $model], 1, "rbf translated train s=$solver");
    run_cmd([$PREDICT, $data, $model, $output], 1, "rbf translated predict s=$solver");
    my @translated = parse_predictions($output);

    check_labels(\@translated, [1, -1], "rbf translated s=$solver");
    for my $i (0 .. $#centered) {
        ok(abs($centered[$i][1] - $translated[$i][1]) < 1e-12, "rbf translation invariance s=$solver row=$i");
    }
}

# 5. rbf distance handles unshared sparse coordinates
write_file(
    $model,
    "solver 0\nkernel 1\ngamma 0.1\nbias 0\nlabels 1 -1\n" .
    "nr_sv 1\nmax_dim 3\nSV\n1 1:2 3:4\n"
);
write_file($data, "1 1:3 2:2\n1 3:5\n1 4:1\n1\n");
run_cmd([$PREDICT, $data, $model, $output], 1, "rbf unshared sparse predict");
my @sparse_preds = parse_predictions($output);
my @expected_dists = (21, 5, 21, 20);
for my $i (0 .. $#expected_dists) {
    my $expected_val = exp(-0.1 * $expected_dists[$i]);
    ok(abs($sparse_preds[$i][1] - $expected_val) < 1e-14, "rbf unshared distance row=$i");
}

# 6. rbf roundtrip with overflowing norms
write_file($data, "1 1:1e200 2:1\n-1 1:1e200 2:-1\n");
run_cmd([$TRAIN, "-t", 1, $data, $model], 1, "rbf huge norm train");
run_cmd([$PREDICT, $data, $model, $output], 1, "rbf huge norm predict");
my @huge_preds = parse_predictions($output);
check_labels(\@huge_preds, [1, -1], "rbf huge norm");

# 7. prediction preserves distinct large labels
write_file($data, "1000001 1:1\n1000000 1:-1\n");
run_cmd([$TRAIN, $data, $model], 1, "large labels train");
run_cmd([$PREDICT, $data, $model, $output], 1, "large labels predict");
my @large_preds = parse_predictions($output);
check_labels(\@large_preds, [1000001, 1000000], "large labels");

# 8. invalid numeric data is rejected
my @invalid_rows = (
    "1 1:garbage", "1 1:", "1 1:2foo", "1foo 1:1", "nan 1:1",
    "inf 1:1", "1 1:nan", "1 1:inf", "1 1:1e999", "1 0:1",
    "1 2:1 1:1", "1 1:1 1:2", "1 2147483648:1", "1 1:2:3",
);
for my $row (@invalid_rows) {
    unlink $model;
    write_file($data, "$row\n-1 1:-1\n");
    run_cmd([$TRAIN, $data, $model], 0, "reject invalid row: $row");
    ok(!-e $model, "no model written for invalid row: $row");
}
run_cmd([$TRAIN, $root, $model], 0, "reject directory as input");

# 9. invalid models are rejected
(my $rbf_header = $HEADER) =~ s/kernel 0/kernel 1/;
$rbf_header =~ s/gamma 0/gamma 1/;
my @invalid_models = (
    "",
    $HEADER,
    $HEADER . "w\n",
    $HEADER . "w 1\nmax_dim 2\n",
    $HEADER . "w 1\nw 2\n",
    $HEADER . "w nan\n",
    $HEADER . "w 1garbage\n",
    $HEADER . "unknown 1\nw 1\n",
    ($HEADER =~ s/max_dim 1/max_dim -1/r) . "w 1\n",
    ($HEADER =~ s/max_dim 1/max_dim 2147483648/r) . "w 1\n",
    ($HEADER =~ s/labels 1 -1\n//r) . "w 1\n",
    $rbf_header,
    $rbf_header . "SV\n1 1:1\n",
    $rbf_header . "SV\n1 1:garbage\n-1 1:-1\n",
    $rbf_header . "SV\n1 1:1 1:2\n-1 1:-1\n",
    $rbf_header . "SV\n1 -1:1\n-1 1:-1\n",
);
write_file($data, $SIMPLE);
for my $idx (0 .. $#invalid_models) {
    write_file($model, $invalid_models[$idx]);
    run_cmd([$PREDICT, $data, $model, $output], 0, "reject invalid model #$idx");
}

# 10. zero feature and zero support models roundtrip
write_file($data, "1\n-1\n");
for my $kernel (0, 1) {
    run_cmd([$TRAIN, "-t", $kernel, $data, $model], 1, "zero feature train t=$kernel");
    run_cmd([$PREDICT, $data, $model, $output], 1, "zero feature predict t=$kernel");
    my @preds = parse_predictions($output);
    is(scalar(@preds), 2, "zero feature count t=$kernel");
}
write_file(
    $model,
    "solver 0\nkernel 1\ngamma 1\nbias -1\nlabels 1 -1\n" .
    "nr_sv 0\nmax_dim 0\n"
);
run_cmd([$PREDICT, $data, $model, $output], 1, "zero support predict");
my @zero_sv_preds = parse_predictions($output);
check_labels(\@zero_sv_preds, [-1, -1], "zero support");

# 11. long sparse rows roundtrip
my $pos = join(" ", map { "$_:1.00000001" } 1 .. 150);
my $neg = join(" ", map { "$_:-1.00000001" } 1 .. 150);
write_file($data, "1 $pos\n-1 $neg");
for my $kernel (0, 1) {
    run_cmd([$TRAIN, "-t", $kernel, $data, $model], 1, "long row train t=$kernel");
    run_cmd([$PREDICT, $data, $model, $output], 1, "long row predict t=$kernel");
    my @preds = parse_predictions($output);
    check_labels(\@preds, [1, -1], "long row t=$kernel");
}

# 12. default model path does not overwrite long input
my $path_max = pathconf($root, _PC_PATH_MAX) // 4096;
SKIP: {
    skip "filesystem PATH_MAX < 1030", 3 if $path_max < 1030;
    my $parent = $root;
    while (1023 - length($parent) - 1 > 200) {
        $parent = File::Spec->catdir($parent, "d" x 180);
        mkdir $parent or die "mkdir failed: $!";
    }
    my $long_path = File::Spec->catfile($parent, "x" x (1023 - length($parent) - 1));
    is(length($long_path), 1023, "constructed 1023-byte path");
    write_file($long_path, $SIMPLE);
    run_cmd([$TRAIN, $long_path], 1, "train with default model path on 1023-byte input");
    is(read_file($long_path), $SIMPLE, "long input file was not overwritten");
    ok(-f "$long_path.model", "long input .model file created");
}

# 13. output open failures are errors
write_file($data, $SIMPLE);
my $unavailable = File::Spec->catfile($root, "missing", "output");
run_cmd([$TRAIN, $data, $unavailable], 0, "train fails on unavailable model path");
ok(!-e $unavailable, "unavailable model path not created");
run_cmd([$TRAIN, $data, $model], 1, "train valid model for output failure tests");
run_cmd([$PREDICT, $data, $model, $unavailable], 0, "predict fails on unavailable output path");
ok(!-e $unavailable, "unavailable output path not created");

# 14. output write failures are errors
SKIP: {
    skip "requires /dev/full", 3 unless -e "/dev/full";
    my ($stdout) = run_cmd([$TRAIN, $data, "/dev/full"], 0, "train fails writing to /dev/full");
    unlike($stdout, qr/Model saved/, "no Model saved message on /dev/full failure");
    run_cmd([$PREDICT, $data, $model, "/dev/full"], 0, "predict fails writing to /dev/full");
}

# 15. support-vector threshold filters numerical dust on SPG and ADMM
write_file($data, "1 1:1\n1 1:10\n1 1:20\n-1 1:-1\n-1 1:-10\n-1 1:-20\n");
for my $case (
    ["-s", 0, "-t", 0, "-m", 3],
    ["-s", 0, "-t", 0, "-m", 4],
    ["-s", 1, "-t", 0, "-e", "1e-7"],
) {
    my $label = join(" ", @$case);
    my ($stdout) = run_cmd([$TRAIN, @$case, $data, $model], 1, "sv threshold linear train ($label)");
    like($stdout, qr/support vectors: 2\)/, "sv threshold filters non-SV dust ($label)");
    like(read_file($model), qr/\bnr_sv 2\n/, "saved model records nr_sv 2 ($label)");
}

write_file($data, "1 1:0.5\n1 1:1.5\n-1 1:-0.5\n-1 1:-1.5\n");
for my $case (
    ["-s", 0, "-t", 1, "-g", "0.25", "-c", 10, "-m", 4],
    ["-s", 1, "-t", 1, "-g", "0.25", "-c", 10, "-e", "1e-7"],
) {
    my $label = join(" ", @$case);
    my ($stdout) = run_cmd([$TRAIN, @$case, $data, $model], 1, "sv threshold rbf train ($label)");
    like($stdout, qr/support vectors: 2\)/, "sv threshold filters rbf non-SV dust ($label)");
    like(read_file($model), qr/\bnr_sv 2\n/, "saved rbf model records nr_sv 2 ($label)");
}

# 16. scale-invariant Gram jitter preserves tiny-norm and huge-norm linear margins
write_file($data, "1 1:1e-6\n-1 1:-1e-6\n");
for my $case (
    ["-s", 0, "-b", 0, "-c", "1e14", "-e", "1e-9"],
    ["-s", 1, "-c", "1e14", "-e", "1e-9"],
) {
    my $label = join(" ", @$case);
    run_cmd([$TRAIN, @$case, $data, $model], 1, "tiny-norm linear train ($label)");
    run_cmd([$PREDICT, $data, $model, $output], 1, "tiny-norm linear predict ($label)");
    my @preds = parse_predictions($output);
    check_labels(\@preds, [1, -1], "tiny-norm linear ($label)");
    ok(abs($preds[0][1] - 1.0) < 1e-3 && abs($preds[1][1] + 1.0) < 1e-3,
       "tiny-norm linear margin undistorted ($label)")
        or diag("got margins: $preds[0][1], $preds[1][1]");
}

write_file($data, "1 1:1e6\n-1 1:-1e6\n");
for my $case (
    ["-s", 0, "-b", 0, "-e", "1e-9"],
    ["-s", 1, "-e", "1e-9"],
) {
    my $label = join(" ", @$case);
    run_cmd([$TRAIN, @$case, $data, $model], 1, "huge-norm linear train ($label)");
    run_cmd([$PREDICT, $data, $model, $output], 1, "huge-norm linear predict ($label)");
    my @preds = parse_predictions($output);
    check_labels(\@preds, [1, -1], "huge-norm linear ($label)");
    ok(abs($preds[0][1] - 1.0) < 1e-3 && abs($preds[1][1] + 1.0) < 1e-3,
       "huge-norm linear margin accurate ($label)")
        or diag("got margins: $preds[0][1], $preds[1][1]");
}

# 17. upfront SvmParam and CLI option validation
write_file($data, $SIMPLE);
for my $bad_args (
    ["-c", "0"],
    ["-c", "-1"],
    ["-c", "nan"],
    ["-c", "inf"],
    ["-e", "0"],
    ["-e", "-1e-4"],
    ["-e", "nan"],
    ["-e", "inf"],
    ["-i", "0"],
    ["-i", "-5"],
    ["-b", "-0.1"],
    ["-b", "nan"],
    ["-b", "inf"],
    ["-b", "1e200"],
    ["-g", "-0.5"],
    ["-g", "nan"],
    ["-g", "inf"],
    ["-s", "-1"],
    ["-s", "2"],
    ["-t", "-1"],
    ["-t", "2"],
    ["-m", "-1"],
    ["-m", "5"],
    ["-s", "garbage"],
    ["-b", "garbage"],
    ["-c", "1foo"],
    ["-sfoo", "0"],
) {
    unlink $model;
    my $label = join(" ", @$bad_args);
    run_cmd([$TRAIN, @$bad_args, $data, $model], 0, "reject invalid param ($label)");
    ok(!-e $model, "no model written for invalid param ($label)");
}

done_testing();
